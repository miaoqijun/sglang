use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(command: &mut Command) {
    let status = command.status().expect("failed to start C++ build command");
    if !status.success() {
        panic!("C++ build command failed: {command:?}");
    }
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let core = manifest.join("../python/sglang/jit_kernel/csrc/ngram_corpus");
    let bridge = manifest.join("cpp/ngram_bridge.cpp");
    let compiler = env::var("CXX").unwrap_or_else(|_| "c++".to_string());

    let sources = [
        bridge,
        core.join("ngram.cpp"),
        core.join("trie.cpp"),
        core.join("result.cpp"),
        core.join("suffix_automaton.cpp"),
    ];
    let mut objects = Vec::new();
    for (index, source) in sources.iter().enumerate() {
        let object = out.join(format!("ngram_{index}.o"));
        run(Command::new(&compiler)
            .arg("-std=c++20")
            .arg("-O3")
            .arg("-DNDEBUG")
            .arg("-fPIC")
            .arg("-pthread")
            .arg("-include")
            .arg("cstddef")
            .arg("-I")
            .arg(&core)
            .arg("-c")
            .arg(source)
            .arg("-o")
            .arg(&object));
        objects.push(object);
    }

    let archive = out.join("libngram_core.a");
    let mut ar = Command::new("ar");
    ar.arg("crs").arg(&archive);
    for object in &objects {
        ar.arg(object);
    }
    run(&mut ar);

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=ngram_core");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!(
        "cargo:rerun-if-changed={}",
        Path::new("cpp/ngram_bridge.cpp").display()
    );
    for source in sources.iter().skip(1) {
        println!("cargo:rerun-if-changed={}", source.display());
    }
}
