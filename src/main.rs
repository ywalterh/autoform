use std::error::Error;
use std::fs;
use std::path::Path;

fn main() {
    // interesting patterns to manipulate filename
    std::process::exit(real_main());
}

fn real_main() -> i32 {
    // Parse input arguments
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <filename>", args[0]);
        return 1;
    }

    let file_name = Path::new(&*args[1]);
    println!("Parsing {}", file_name.display());
    match unzip_odf(file_name) {
        Err(err) => {
            println!("Error: {}", err);
            return 1;
        }
        Ok(()) => (),
    };
    return 0;
}

fn unzip_odf(file_name: &Path) -> Result<(), Box<dyn Error>> {
    let file = fs::File::open(file_name)?;
    let mut archive = zip::ZipArchive::new(file)?;

    for i in 0..archive.len() {
        let file = archive.by_index(i).unwrap();
        println!(
            "File #{} : {}",
            i,
            file.sanitized_name().as_path().display()
        );
    }

    return Ok(());
}
