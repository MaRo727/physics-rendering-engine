use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    compile_shader("src/shaders/raygen.rgen",      out_dir.join("raygen.rgen.spv"));
    compile_shader("src/shaders/miss.rmiss",       out_dir.join("miss.rmiss.spv"));
    compile_shader("src/shaders/shadow.rmiss",     out_dir.join("shadow.rmiss.spv"));
    compile_shader("src/shaders/closesthit.rchit", out_dir.join("closesthit.rchit.spv"));
}

fn compile_shader(input: &str, output: PathBuf) {
    println!("cargo:rerun-if-changed={input}");
    let status = Command::new("glslc")
        .args(["--target-env=vulkan1.3", input, "-o", output.to_str().unwrap()])
        .status()
        .expect("glslc not found — install the Vulkan SDK");
    assert!(status.success(), "glslc failed on {input}");
}
