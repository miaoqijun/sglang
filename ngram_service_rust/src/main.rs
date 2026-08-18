use std::collections::HashSet;
use std::env;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::io;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const MAGIC: &[u8; 4] = b"NGRS";
const VERSION: u8 = 1;
const FRAME_HEADER_SIZE: usize = 10;
const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;
const OP_BATCH_GET: u8 = 1;
const OP_BATCH_PUT: u8 = 2;
const OP_ERASE_MATCH_STATE: u8 = 3;
const OP_RESPONSE_BIT: u8 = 0x80;
const OP_ERROR: u8 = 0xff;

extern "C" {
    fn ngram_last_error() -> *const c_char;
    fn ngram_create(
        capacity: u64,
        max_trie_depth: u64,
        min_bfs_breadth: u64,
        max_bfs_breadth: u64,
        draft_token_num: u64,
        match_type: c_int,
    ) -> *mut c_void;
    fn ngram_destroy(handle: *mut c_void);
    fn ngram_batch_put(
        handle: *mut c_void,
        flat_tokens: *const i32,
        token_count: usize,
        offsets: *const i64,
        batch_size: usize,
        wait_for_visibility: c_int,
    ) -> c_int;
    fn ngram_batch_get(
        handle: *mut c_void,
        state_ids: *const i64,
        total_lens: *const i64,
        flat_tokens: *const i32,
        token_count: usize,
        offsets: *const i64,
        batch_size: usize,
        output_tokens: *mut i32,
        output_mask: *mut u8,
    ) -> c_int;
    fn ngram_erase_match_state(handle: *mut c_void, state_ids: *const i64, count: usize) -> c_int;
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn native_error() -> io::Error {
    unsafe {
        let raw = ngram_last_error();
        let message = if raw.is_null() {
            "unknown native NGRAM error".to_string()
        } else {
            CStr::from_ptr(raw).to_string_lossy().into_owned()
        };
        io::Error::other(message)
    }
}

struct Corpus {
    raw: *mut c_void,
    draft_token_num: usize,
}

unsafe impl Send for Corpus {}
unsafe impl Sync for Corpus {}

impl Corpus {
    fn create(config: &Config) -> io::Result<Self> {
        let raw = unsafe {
            ngram_create(
                config.capacity,
                config.max_trie_depth,
                config.min_bfs_breadth,
                config.max_bfs_breadth,
                config.draft_token_num as u64,
                if config.match_type == "BFS" { 0 } else { 1 },
            )
        };
        if raw.is_null() {
            return Err(native_error());
        }
        Ok(Self {
            raw,
            draft_token_num: config.draft_token_num,
        })
    }

    fn batch_put(&self, tokens: &[i32], offsets: &[i64], wait: bool) -> io::Result<()> {
        let rc = unsafe {
            ngram_batch_put(
                self.raw,
                tokens.as_ptr(),
                tokens.len(),
                offsets.as_ptr(),
                offsets.len() - 1,
                if wait { 1 } else { 0 },
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(native_error())
        }
    }

    fn batch_get(
        &self,
        state_ids: &[i64],
        total_lens: &[i64],
        tokens: &[i32],
        offsets: &[i64],
    ) -> io::Result<(Vec<i32>, Vec<u8>)> {
        let batch_size = state_ids.len();
        let mut output_tokens = vec![0i32; batch_size * self.draft_token_num];
        let mut output_mask = vec![0u8; batch_size * self.draft_token_num * self.draft_token_num];
        let rc = unsafe {
            ngram_batch_get(
                self.raw,
                state_ids.as_ptr(),
                total_lens.as_ptr(),
                tokens.as_ptr(),
                tokens.len(),
                offsets.as_ptr(),
                batch_size,
                output_tokens.as_mut_ptr(),
                output_mask.as_mut_ptr(),
            )
        };
        if rc == 0 {
            Ok((output_tokens, output_mask))
        } else {
            Err(native_error())
        }
    }

    fn erase(&self, state_ids: &[i64]) -> io::Result<()> {
        let rc = unsafe { ngram_erase_match_state(self.raw, state_ids.as_ptr(), state_ids.len()) };
        if rc == 0 {
            Ok(())
        } else {
            Err(native_error())
        }
    }
}

impl Drop for Corpus {
    fn drop(&mut self) {
        unsafe { ngram_destroy(self.raw) }
    }
}

#[derive(Clone)]
struct Config {
    host: String,
    port: u16,
    capacity: u64,
    max_trie_depth: u64,
    min_bfs_breadth: u64,
    max_bfs_breadth: u64,
    draft_token_num: usize,
    match_type: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 31291,
            capacity: 10_000_000,
            max_trie_depth: 18,
            min_bfs_breadth: 1,
            max_bfs_breadth: 10,
            draft_token_num: 12,
            match_type: "BFS".to_string(),
        }
    }
}

impl Config {
    fn parse() -> io::Result<Self> {
        let mut config = Self::default();
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| invalid(format!("missing value for {flag}")))?;
            match flag.as_str() {
                "--host" => config.host = value,
                "--port" => config.port = value.parse().map_err(|_| invalid("invalid --port"))?,
                "--capacity" => {
                    config.capacity = value.parse().map_err(|_| invalid("invalid --capacity"))?
                }
                "--max-trie-depth" => {
                    config.max_trie_depth = value
                        .parse()
                        .map_err(|_| invalid("invalid --max-trie-depth"))?
                }
                "--min-bfs-breadth" => {
                    config.min_bfs_breadth = value
                        .parse()
                        .map_err(|_| invalid("invalid --min-bfs-breadth"))?
                }
                "--max-bfs-breadth" => {
                    config.max_bfs_breadth = value
                        .parse()
                        .map_err(|_| invalid("invalid --max-bfs-breadth"))?
                }
                "--draft-token-num" => {
                    config.draft_token_num = value
                        .parse()
                        .map_err(|_| invalid("invalid --draft-token-num"))?
                }
                "--match-type" if value == "BFS" || value == "PROB" => config.match_type = value,
                "--match-type" => return Err(invalid("--match-type must be BFS or PROB")),
                _ => return Err(invalid(format!("unknown argument {flag}"))),
            }
        }
        Ok(config)
    }
}

fn u32_at(data: &[u8], offset: usize) -> io::Result<u32> {
    let bytes: [u8; 4] = data
        .get(offset..offset + 4)
        .ok_or_else(|| invalid("truncated u32"))?
        .try_into()
        .unwrap();
    Ok(u32::from_le_bytes(bytes))
}

fn parse_i64s(data: &[u8], offset: usize, count: usize) -> io::Result<Vec<i64>> {
    let raw = data
        .get(offset..offset + count * 8)
        .ok_or_else(|| invalid("truncated int64 array"))?;
    Ok(raw
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_i32s(data: &[u8], offset: usize, count: usize) -> io::Result<Vec<i32>> {
    let raw = data
        .get(offset..offset + count * 4)
        .ok_or_else(|| invalid("truncated int32 array"))?;
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn validate_offsets(offsets: &[i64], token_count: usize) -> io::Result<()> {
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] != token_count as i64 {
        return Err(invalid("invalid CSR offsets"));
    }
    if offsets
        .windows(2)
        .any(|pair| pair[0] < 0 || pair[0] > pair[1])
    {
        return Err(invalid("CSR offsets must be nonnegative and nondecreasing"));
    }
    Ok(())
}

async fn read_frame(stream: &mut TcpStream) -> io::Result<Option<(u8, Vec<u8>)>> {
    let mut header = [0u8; FRAME_HEADER_SIZE];
    match stream.read_exact(&mut header).await {
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error),
    }
    if &header[0..4] != MAGIC || header[4] != VERSION {
        return Err(invalid("invalid NGRAM frame header"));
    }
    let size = u32::from_le_bytes(header[6..10].try_into().unwrap()) as usize;
    if size > MAX_FRAME_BYTES {
        return Err(invalid("frame is too large"));
    }
    let mut payload = vec![0u8; size];
    stream.read_exact(&mut payload).await?;
    Ok(Some((header[5], payload)))
}

async fn write_frame(stream: &mut TcpStream, opcode: u8, payload: &[u8]) -> io::Result<()> {
    let mut frame = Vec::with_capacity(FRAME_HEADER_SIZE + payload.len());
    frame.extend_from_slice(MAGIC);
    frame.push(VERSION);
    frame.push(opcode);
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(payload);
    stream.write_all(&frame).await
}

fn namespaced_states(local_states: Vec<i64>, session_id: u32) -> io::Result<Vec<i64>> {
    let prefix = (session_id as i64) << 32;
    local_states
        .into_iter()
        .map(|state| {
            if !(0..=u32::MAX as i64).contains(&state) {
                Err(invalid("local state id must fit in u32"))
            } else {
                Ok(prefix | state)
            }
        })
        .collect()
}

async fn dispatch(
    stream: &mut TcpStream,
    corpus: &Corpus,
    session_id: u32,
    known_states: &mut HashSet<i64>,
    opcode: u8,
    payload: &[u8],
) -> io::Result<()> {
    match opcode {
        OP_BATCH_PUT => {
            if payload.len() < 16 {
                return Err(invalid("truncated batch_put header"));
            }
            let batch_size = u32_at(payload, 0)? as usize;
            let token_count = u32_at(payload, 4)? as usize;
            let wait = payload[8];
            if batch_size == 0 || wait > 1 {
                return Err(invalid("invalid batch_put header"));
            }
            let offsets_offset = 16;
            let tokens_offset = offsets_offset + (batch_size + 1) * 8;
            let expected = tokens_offset + token_count * 4;
            if payload.len() != expected {
                return Err(invalid("invalid batch_put payload size"));
            }
            let offsets = parse_i64s(payload, offsets_offset, batch_size + 1)?;
            let tokens = parse_i32s(payload, tokens_offset, token_count)?;
            validate_offsets(&offsets, token_count)?;
            corpus.batch_put(&tokens, &offsets, wait != 0)?;
            if wait != 0 {
                write_frame(
                    stream,
                    opcode | OP_RESPONSE_BIT,
                    &(batch_size as u64).to_le_bytes(),
                )
                .await
            } else {
                Ok(())
            }
        }
        OP_BATCH_GET => {
            if payload.len() < 8 {
                return Err(invalid("truncated batch_get header"));
            }
            let batch_size = u32_at(payload, 0)? as usize;
            let token_count = u32_at(payload, 4)? as usize;
            if batch_size == 0 {
                return Err(invalid("batch_get batch size must be positive"));
            }
            let state_offset = 8;
            let lens_offset = state_offset + batch_size * 8;
            let offsets_offset = lens_offset + batch_size * 8;
            let tokens_offset = offsets_offset + (batch_size + 1) * 8;
            let expected = tokens_offset + token_count * 4;
            if payload.len() != expected {
                return Err(invalid("invalid batch_get payload size"));
            }
            let states =
                namespaced_states(parse_i64s(payload, state_offset, batch_size)?, session_id)?;
            let total_lens = parse_i64s(payload, lens_offset, batch_size)?;
            let offsets = parse_i64s(payload, offsets_offset, batch_size + 1)?;
            let tokens = parse_i32s(payload, tokens_offset, token_count)?;
            validate_offsets(&offsets, token_count)?;
            known_states.extend(states.iter().copied());
            let (output_tokens, output_mask) =
                corpus.batch_get(&states, &total_lens, &tokens, &offsets)?;

            let mut response = Vec::with_capacity(8 + output_tokens.len() * 4 + output_mask.len());
            response.extend_from_slice(&(batch_size as u32).to_le_bytes());
            response.extend_from_slice(&(corpus.draft_token_num as u32).to_le_bytes());
            for token in output_tokens {
                response.extend_from_slice(&token.to_le_bytes());
            }
            response.extend_from_slice(&output_mask);
            write_frame(stream, opcode | OP_RESPONSE_BIT, &response).await
        }
        OP_ERASE_MATCH_STATE => {
            if payload.len() < 8 {
                return Err(invalid("truncated erase header"));
            }
            let count = u32_at(payload, 0)? as usize;
            if payload.len() != 8 + count * 8 {
                return Err(invalid("invalid erase payload size"));
            }
            let states = namespaced_states(parse_i64s(payload, 8, count)?, session_id)?;
            corpus.erase(&states)?;
            for state in &states {
                known_states.remove(state);
            }
            write_frame(
                stream,
                opcode | OP_RESPONSE_BIT,
                &(count as u64).to_le_bytes(),
            )
            .await
        }
        _ => Err(invalid("unknown opcode")),
    }
}

async fn handle_connection(
    mut stream: TcpStream,
    corpus: Arc<Corpus>,
    session_id: u32,
) -> io::Result<()> {
    stream.set_nodelay(true)?;
    let mut known_states = HashSet::new();
    loop {
        let frame = match read_frame(&mut stream).await {
            Ok(Some(frame)) => frame,
            Ok(None) => break,
            Err(error) => {
                let _ = write_frame(&mut stream, OP_ERROR, error.to_string().as_bytes()).await;
                break;
            }
        };
        if let Err(error) = dispatch(
            &mut stream,
            &corpus,
            session_id,
            &mut known_states,
            frame.0,
            &frame.1,
        )
        .await
        {
            let _ = write_frame(&mut stream, OP_ERROR, error.to_string().as_bytes()).await;
            break;
        }
    }
    if !known_states.is_empty() {
        let states: Vec<_> = known_states.into_iter().collect();
        let _ = corpus.erase(&states);
    }
    Ok(())
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let config = Config::parse()?;
    let corpus = Arc::new(Corpus::create(&config)?);
    let listener = TcpListener::bind((config.host.as_str(), config.port)).await?;
    let next_session = AtomicU32::new(1);
    eprintln!(
        "NGRAM Rust service listening on {}:{} with one shared Trie",
        config.host, config.port
    );

    loop {
        let (stream, _) = listener.accept().await?;
        let session_id = next_session.fetch_add(1, Ordering::Relaxed);
        let corpus = Arc::clone(&corpus);
        tokio::spawn(async move {
            if let Err(error) = handle_connection(stream, corpus, session_id).await {
                eprintln!("connection failed: {error}");
            }
        });
    }
}
