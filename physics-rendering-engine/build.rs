use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    compile_shader("src/shaders/mesh.vert", out_dir.join("mesh.vert.spv"));
    compile_shader("src/shaders/mesh.frag", out_dir.join("mesh.frag.spv"));
}

fn compile_shader(input: &str, output: PathBuf) {
    println!("cargo:rerun-if-changed={input}");
    let status = Command::new("glslc")
        .args([input, "-o", output.to_str().unwrap()])
        .status()
        .expect("glslc not found — install the Vulkan SDK");
    assert!(status.success(), "glslc failed on {input}");
}
