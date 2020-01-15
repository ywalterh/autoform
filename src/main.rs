use quick_xml::events::Event;
use quick_xml::Reader;
use std::error::Error;
use std::fs;
use std::io::prelude::*;
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
    let mut file = archive.by_name("content.xml")?;

    // print out the entire string of xml file
    let mut xml_content_buffer = String::new();
    file.read_to_string(&mut xml_content_buffer)?;

    let mut reader = Reader::from_str(xml_content_buffer.as_str());
    reader.trim_text(true);

    let mut count = 0;
    let mut txt = Vec::new();
    let mut buf = Vec::new();

    // The `Reader` does not implement `Iterator` because it outputs borrowed data (`Cow`s)
    loop {
        match reader.read_event(&mut buf) {
            Ok(Event::Start(ref e)) => {
                println!("tag name is {}", std::str::from_utf8(e.name()).unwrap());
                println!(
                    "attributes values: {:?}",
                    e.attributes().map(|a| std::str::from_utf8(a.unwrap().key).unwrap()).collect::<Vec<_>>()
                );
                match e.name() {
                    b"tag1" => println!(
                        "attributes values: {:?}",
                        e.attributes().map(|a| a.unwrap().value).collect::<Vec<_>>()
                    ),
                    b"text:list-style" => count += 1,
                    _ => (),
                }
            }
            Ok(Event::Text(e)) => txt.push(e.unescape_and_decode(&reader).unwrap()),
            Ok(Event::Eof) => break, // exits the loop when reaching end of file
            Err(e) => panic!("Error at position {}: {:?}", reader.buffer_position(), e),
            _ => (), // There are several other `Event`s we do not consider here
        }

        // if we don't keep a borrow elsewhere, we can clear the buffer to keep memory usage low
        buf.clear();
    }

    println!("Found text:list-style {} times", count);
    return Ok(());
}
