use crate::config::CacheConfig;
use serde::{Deserialize,Serialize};
use sha2::{Digest,Sha256};
use std::{io::{Read,Write},path::Path};
pub const SCHEMA:&str="narc-v1:pos3-normal3-view3-light3-albedo3-roughness-metallic:orbit-z:rgbx-f32-le";
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq,Eq)]
pub struct Namespace {pub teacher_version:u32,pub scene_version:u32,pub material_version:u32,pub quantization_version:u32}
impl Default for Namespace {fn default()->Self{Self{teacher_version:1,scene_version:1,material_version:1,quantization_version:1}}}
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq,Eq)]
pub struct Metadata {pub schema_hash:String,pub teacher_weights_hash:String,pub namespace:Namespace,pub quantization:CacheConfig,pub valid_objects:[bool;6],pub payload_dimension:usize,pub payload_bytes:usize}
impl Metadata {
    pub fn new(c:&CacheConfig,hash:String)->anyhow::Result<Self>{Ok(Self{schema_hash:format!("{:x}",Sha256::digest(SCHEMA.as_bytes())),teacher_weights_hash:hash,namespace:Namespace::default(),quantization:c.clone(),valid_objects:[true;6],payload_dimension:4,payload_bytes:c.bytes()?})}
    pub fn invalidate(&mut self,object:usize)->anyhow::Result<()> {anyhow::ensure!(object<6,"object ID outside namespace");self.valid_objects[object]=false;Ok(())}
}
pub fn save(path:&Path,meta:&Metadata,data:&[f32])->anyhow::Result<()> {
    anyhow::ensure!(data.len().checked_mul(4)==Some(meta.payload_bytes),"payload size differs from metadata");
    if let Some(parent)=path.parent(){std::fs::create_dir_all(parent)?;}
    let mut file=std::io::BufWriter::new(std::fs::File::create(path)?);
    let header=serde_json::to_vec(meta)?;file.write_all(b"NARCACHE")?;file.write_all(&1u32.to_le_bytes())?;file.write_all(&(header.len() as u32).to_le_bytes())?;file.write_all(&header)?;
    let mut checksum=Sha256::new();checksum.update(&header);
    for chunk in data.chunks(16384){let bytes:Vec<u8>=chunk.iter().flat_map(|x|x.to_le_bytes()).collect();checksum.update(&bytes);file.write_all(&bytes)?;}
    file.write_all(&checksum.finalize())?;file.flush()?;Ok(())
}
pub fn load(path:&Path,expected:&Metadata)->anyhow::Result<(Metadata,Vec<f32>)> {
    let mut file=std::io::BufReader::new(std::fs::File::open(path)?);let mut magic=[0u8;8];file.read_exact(&mut magic)?;anyhow::ensure!(&magic==b"NARCACHE","invalid cache magic");
    let mut word=[0u8;4];file.read_exact(&mut word)?;anyhow::ensure!(u32::from_le_bytes(word)==1,"unsupported cache version");file.read_exact(&mut word)?;let len=u32::from_le_bytes(word) as usize;anyhow::ensure!(len<=65536,"oversized cache header");
    let mut header=vec![0;len];file.read_exact(&mut header)?;let meta:Metadata=serde_json::from_slice(&header)?;
    let mut comparable=meta.clone();comparable.valid_objects=expected.valid_objects;
    anyhow::ensure!(comparable==*expected,"cache schema, teacher, namespace or quantization mismatch");
    anyhow::ensure!(file.get_ref().metadata()?.len()==16+len as u64+meta.payload_bytes as u64+32,"cache length mismatch");
    let mut checksum=Sha256::new();checksum.update(&header);let mut data=Vec::with_capacity(meta.payload_bytes/4);let mut chunk=vec![0u8;65536];let mut remaining=meta.payload_bytes;
    while remaining>0 {let n=remaining.min(chunk.len());file.read_exact(&mut chunk[..n])?;checksum.update(&chunk[..n]);for b in chunk[..n].chunks_exact(4){data.push(f32::from_le_bytes(b.try_into()?));}remaining-=n;}
    let mut stored=[0u8;32];file.read_exact(&mut stored)?;anyhow::ensure!(checksum.finalize().as_slice()==stored,"cache checksum mismatch");
    anyhow::ensure!(data.iter().all(|v|v.is_finite()),"non-finite cache payload");Ok((meta,data))
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn roundtrip_and_rejection()->anyhow::Result<()> {
        let dir=std::env::temp_dir().join(format!("narc-test-{}",std::process::id()));std::fs::create_dir_all(&dir)?;let path=dir.join("test.narc");
        let c=CacheConfig{position_grid:[1;3],view_bins:1,light_bins:1,..Default::default()};let meta=Metadata::new(&c,"teacher".into())?;let data=vec![0.25;24];save(&path,&meta,&data)?;assert_eq!(load(&path,&meta)?.1,data);
        let mut wrong=meta.clone();wrong.namespace.material_version+=1;assert!(load(&path,&wrong).is_err());wrong=meta.clone();wrong.schema_hash="wrong".into();assert!(load(&path,&wrong).is_err());
        let mut bytes=std::fs::read(&path)?;let n=bytes.len();bytes[n-33]^=1;std::fs::write(&path,bytes)?;assert!(load(&path,&meta).is_err());std::fs::remove_file(path)?;std::fs::remove_dir(dir)?;Ok(())
    }
    #[test]fn partial_invalidation(){let mut m=Metadata::new(&CacheConfig::default(),"teacher".into()).unwrap();m.invalidate(2).unwrap();assert_eq!(m.valid_objects,[true,true,false,true,true,true]);assert!(m.invalidate(6).is_err());}
}
