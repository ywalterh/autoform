use quick_xml::events::{BytesEnd, BytesStart, Event};
use quick_xml::Reader;
use quick_xml::Writer;
use std::borrow::Cow;
use std::error::Error;
use std::fs;
use std::io::prelude::*;
use std::io::Cursor;
use std::path::Path;
use zip::write::FileOptions;

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
    let zipfile = fs::File::open(file_name)?;
    let mut archive = zip::ZipArchive::new(&zipfile)?;
    let mut file = archive.by_name("content.xml")?;

    // print out the entire string of xml file
    let mut xml_content_buffer = String::new();
    file.read_to_string(&mut xml_content_buffer)?;

    let mut reader = Reader::from_str(xml_content_buffer.as_str());
    reader.trim_text(true);

    let mut writer = Writer::new(Cursor::new(Vec::new()));

    //let mut txt = Vec::new();
    let mut buf = Vec::new();

    // The `Reader` does not implement `Iterator` because it outputs borrowed data (`Cow`s)
    loop {
        // let's match attributes with svg: ... and then modify it in place
        // hopefully it's doable
        match reader.read_event(&mut buf) {
            Ok(Event::Start(ref e)) => {
                // Let's copy it for now, not sure if it's feasible to
                // do in place update
                let mut elem = BytesStart::owned(e.name(), e.name().len());

                // push existing elem along, but perform inplace update
                // if attribute contains svg: like settings
                // assuming them to be cm for now
                elem.extend_attributes(e.attributes().map(|attr| {
                    let mut attribute = attr.unwrap();
                    let key = std::str::from_utf8(attribute.key).unwrap();
                    if key.contains("svg:") {
                        attribute.value =Cow::Borrowed(b"1cm");
                    }

                    return attribute;
                }));

                assert!(writer.write_event(Event::Start(elem)).is_ok());
            }
            Ok(Event::End(ref e)) => {
                assert!(writer
                    .write_event(Event::End(BytesEnd::borrowed(e.name())))
                    .is_ok());
            }
            Ok(Event::Eof) => break, // exits the loop when reaching end of file
            Ok(e) => assert!(writer.write_event(&e).is_ok()),
            Err(e) => panic!("Error at position {}: {:?}", reader.buffer_position(), e),
        }

        // if we don't keep a borrow elsewhere, we can clear the buffer to keep memory usage low
        buf.clear();
    }

    let result = writer.into_inner().into_inner();
    let mut zip = zip::ZipWriter::new(&zipfile);
    let options = FileOptions::default().compression_method(zip::CompressionMethod::Stored).unix_permissions(0o755);
    zip.start_file("content.xml", options)?;
    zip.write_all(&result)?;
    zip.finish()?;
    return Ok(());
}
