use std::error::Error;
use xml::attribute::Attribute;
use xml::reader::XmlEvent;
use xml::EventReader;
// This file is used to extract feature from a parsed XML content tree

// Based on the index
const PRESENTATION_CLASS: &'static [&'static str] = &["title", "subtitle"];
const CHILD_TAG: &'static [&'static str] = &["draw:text-box", "table"];

#[derive(Debug, Default)]
pub struct Feature {
    presentation_class: u8,
    child_tag: u8,
    text_length: u8,
    table_column: u8,
    table_row: u8,
}

fn extra_features(xml_content_buffer: &str) -> Result<Vec<Feature>, Box<dyn Error>> {
    let parser = EventReader::from_str(xml_content_buffer);

    for e in parser {
        match e {
            Ok(XmlEvent::StartElement {
                name, attributes, ..
            }) => {
                let mut attrs: Vec<Attribute> = Vec::new();
                for mut attribute in attributes.iter() {}
                break;
            }
            Err(err) => {
                println!("Error: {}", err);
                break;
            }
            _ => {}
        }
    }

    let mut result: Vec<Feature> = Vec::new();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use crate::feature::extra_features;
    use std::fs::File;
    use std::io::{BufReader, Read};

    #[test]
    fn test_feature_extraction() {
        let file = File::open("test_data/content.xml").unwrap();
        // can we use the same variable name?
        let mut file = BufReader::new(file);
        let mut file_content: String = String::new();
        file.read_to_string(&mut file_content);
        let result = extra_features(file_content.as_str()).unwrap();
        let title = &result[0];
        let sub_title = &result[1];
        let table = &result[2];

        assert_eq!(title.presentation_class, 1);
        assert_eq!(title.child_tag, 1);
        assert_eq!(title.text_length, 7);
        assert_eq!(title.table_column, 0);
        assert_eq!(title.table_row, 0);

        assert_eq!(sub_title.presentation_class, 2);
        assert_eq!(sub_title.child_tag, 1);
        assert_eq!(sub_title.text_length, 10);
        assert_eq!(sub_title.table_column, 0);
        assert_eq!(sub_title.table_row, 0);

        assert_eq!(table.presentation_class, 0);
        assert_eq!(table.child_tag, 1);
        assert_eq!(table.text_length, 20);
        assert_eq!(sub_title.table_column, 7);
        assert_eq!(sub_title.table_row, 5);
    }
}
