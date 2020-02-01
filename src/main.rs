extern crate rand;

use rand::Rng;
use std::borrow::Cow;
use std::error::Error;
use std::fs;
use std::io::prelude::*;
use std::path::Path;
use xml::attribute::Attribute;
use xml::reader::{EventReader, XmlEvent};
use xml::writer::EmitterConfig;
use xml::writer::XmlEvent as wXmlEvent;
use zip::write::FileOptions;

// import neural network module
mod nn;

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

    // With the resulting content.xml, now it's time to duplicate files from the zips
    let path = std::path::Path::new("./updated.odp");
    let file = std::fs::File::create(&path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let options = FileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .unix_permissions(0o755);

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        if file.name() == "content.xml" {
            let mut xml_content_buffer = String::new();
            file.read_to_string(&mut xml_content_buffer)?;
            zip.start_file("content.xml", options)?;
            zip.write_all(&update_content_xml(&xml_content_buffer)?)?;
        } else {
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            zip.start_file(file.name(), options)?;
            zip.write(&buf)?;
        }
    }

    zip.finish()?;
    return Ok(());
}

// Insead of doing in place update a better approach is to build
// data binding for each document format and then update it to
// save into odp files
fn update_content_xml(xml_content_buffer: &str) -> Result<std::vec::Vec<u8>, Box<dyn Error>> {
    let mut rng = rand::thread_rng();
    let parser = EventReader::from_str(xml_content_buffer);
    let mut target: Vec<XmlEvent> = Vec::new();
    for e in parser {
        match e {
            Err(e) => {
                println!("Error: {}", e);
                break;
            }
            // grab the rest
            elem => match elem {
                Ok(elem_real) => {
                    target.push(elem_real);
                }
                Err(_) => {}
            },
        }
    }

    let mut content_target: Vec<u8> = Vec::new();
    let mut writer = EmitterConfig::new()
        .perform_indent(true)
        .create_writer(&mut content_target);

    //@Cleanup setup result to handle Err
    for t in target {
        match t {
            XmlEvent::StartElement {
                name,
                mut attributes,
                namespace,
            } => {
                // Randomize attribute with svg
                let mut attrs: Vec<Attribute> = Vec::new();
                for mut attribute in attributes.iter_mut() {
                    // copy the value
                    let copy_actual_value: String = String::from(&attribute.value);

                    // setup prefix
                    match &attribute.name.prefix {
                        Some(prefix) => {
                            if prefix == "svg" {
                                if copy_actual_value.ends_with("cm") {
                                    let random_s = format!("{}cm", rng.gen_range(1, 10));
                                    attribute.value = random_s;
                                }
                            }
                        }
                        None => {}
                    }

                    attrs.push(attribute.borrow());
                }

                writer.write(wXmlEvent::StartElement {
                    name: name.borrow(),
                    attributes: Cow::Owned(attrs),
                    namespace: Cow::Borrowed(&namespace),
                })?;
            }
            XmlEvent::EndElement { name } => {
                writer.write(wXmlEvent::EndElement {
                    name: Some(name.borrow()),
                })?;
            }
            XmlEvent::StartDocument {
                version,
                encoding,
                standalone,
            } => {
                writer.write(wXmlEvent::StartDocument {
                    version,
                    encoding: Some(&encoding),
                    standalone,
                })?;
            }
            XmlEvent::Characters(c) => {
                writer.write(wXmlEvent::Characters(&c))?;
            }
            XmlEvent::Comment(c) => {
                writer.write(wXmlEvent::Comment(&c))?;
            }
            XmlEvent::CData(c) => {
                writer.write(wXmlEvent::CData(&c))?;
            }
            _ => {}
        }
    }

    Ok(content_target)
}
