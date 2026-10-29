//! Decodes one parquet column chunk of strings: a row group's text column,
//! downloaded on its own with a range request.
//!
//! A chunk's pages carry their own headers, so the only footer metadata needed
//! is what the coordinator sends in [`ParquetChunk`].

use bytes::Bytes;
use dti_protocol::{Codec, ParquetChunk};
use parquet::basic::Compression;
use parquet::column::reader::ColumnReaderImpl;
use parquet::data_type::ByteArrayType;
use parquet::file::metadata::ColumnChunkMetaData;
use parquet::file::serialized_reader::SerializedPageReader;
use parquet::schema::parser::parse_message_type;
use parquet::schema::types::SchemaDescriptor;
use std::sync::Arc;

/// The text of the chunk's kept rows, one document per line.
pub fn text(chunk: &ParquetChunk, raw: Vec<u8>) -> Result<Vec<u8>, String> {
    let err = |e: parquet::errors::ParquetError| format!("decoding {}: {e}", chunk.url);
    let rows = usize::try_from(chunk.num_values).map_err(|_| "too many rows".to_string())?;
    let repetition = if chunk.nullable {
        "optional"
    } else {
        "required"
    };
    let schema = parse_message_type(&format!("message m {{ {repetition} binary text (UTF8); }}"))
        .map_err(err)?;
    let column = SchemaDescriptor::new(Arc::new(schema)).column(0);
    // Offsets are relative to the downloaded chunk, which starts at its
    // dictionary page if it has one.
    let meta = ColumnChunkMetaData::builder(column.clone())
        .set_compression(match chunk.codec {
            Codec::Uncompressed => Compression::UNCOMPRESSED,
            Codec::Snappy => Compression::SNAPPY,
        })
        .set_num_values(chunk.num_values as i64)
        .set_total_compressed_size(raw.len() as i64)
        .set_data_page_offset(chunk.data_page_offset as i64)
        .set_dictionary_page_offset((chunk.data_page_offset > 0).then_some(0))
        .build()
        .map_err(err)?;
    let pages =
        SerializedPageReader::new(Arc::new(Bytes::from(raw)), &meta, rows, None).map_err(err)?;
    let mut reader = ColumnReaderImpl::<ByteArrayType>::new(column, Box::new(pages));

    let mut values = Vec::with_capacity(rows);
    let mut defs = Vec::with_capacity(rows);
    let (read, _, _) = reader
        .read_records(rows, chunk.nullable.then_some(&mut defs), None, &mut values)
        .map_err(err)?;
    if read != rows {
        return Err(format!("{}: expected {rows} rows, got {read}", chunk.url));
    }

    if !chunk.nullable {
        defs = vec![1; rows];
    }
    let mut out = Vec::new();
    let mut values = values.iter();
    for (row, &def) in defs.iter().enumerate() {
        // Nulls have no value: only rows at the max definition level do.
        let value = if def == 1 { values.next() } else { None };
        if let Some(value) = value.filter(|_| keep(chunk, row)) {
            out.extend_from_slice(value.data());
            out.push(b'\n');
        }
    }
    Ok(out)
}

fn keep(chunk: &ParquetChunk, row: usize) -> bool {
    chunk
        .keep
        .as_ref()
        .is_none_or(|mask| mask.get(row / 8).is_some_and(|b| b >> (row % 8) & 1 == 1))
}
