use std::{env, path::PathBuf, process::Command};

use bindgen::CargoCallbacks;
use regex::Regex;

fn main() {
    if cfg!(feature = "cuda-sketch") {
        println!("cargo:rerun-if-changed=cuda");
        println!("cargo:rerun-if-changed=src/cuda_kernel.cu");
        println!("cargo:rerun-if-changed=src/cuda_kernel.h");

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        let cuda_src = PathBuf::from("src/cuda_kernel.cu");
        let ptx_file = out_dir.join("cuda_kmer_hash.ptx");

        let nvcc_status = Command::new("nvcc")
            .arg("-ptx")
            .arg("-o")
            .arg(&ptx_file)
            .arg(&cuda_src)
            .arg("-arch=compute_70")
            .status()
            .unwrap();

        assert!(
            nvcc_status.success(),
            "Failed to compile CUDA source to PTX."
        );

        let bindings = bindgen::Builder::default()
            .header("src/cuda_kernel.h")
            .parse_callbacks(Box::new(CargoCallbacks))
            .no_copy("*")
            .no_debug("*")
            .generate()
            .expect("Unable to generate bindings");

        let generated_bindings = bindings.to_string();

        let pointer_regex = Regex::new(r"\*mut f32").unwrap();
        let modified_bindings =
            pointer_regex.replace_all(&generated_bindings, "CudaSlice<f32>");

        std::fs::write(out_dir.join("bindings.rs"), modified_bindings.as_bytes())
            .expect("Failed to write bindings");
    }
}