use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all="snake_case")]
pub enum Sequence { Static, Camera, CameraLight }
impl Sequence {
    pub fn code(self) -> i32 { match self { Self::Static=>0, Self::Camera=>1, Self::CameraLight=>2 } }
    pub fn angles(self, frame: u32) -> (f32,f32) {
        (if self==Self::Static {0.0} else {frame as f32*0.013}, if self==Self::CameraLight {frame as f32*0.0031} else {0.0})
    }
}
impl std::str::FromStr for Sequence {
    type Err=anyhow::Error;
    fn from_str(s:&str)->Result<Self,Self::Err>{match s.to_ascii_lowercase().replace('-',"_").as_str(){"static"=>Ok(Self::Static),"camera"=>Ok(Self::Camera),"camera_light"=>Ok(Self::CameraLight),_=>anyhow::bail!("unknown sequence {s}")}}
}
pub fn features(p:[f32;3], object:u32, view:f32, light:f32)->[f32;17] {
    let [x,y,z]=p;
    let nx=-0.75*(3.0*x).cos()*(3.0*y).cos();
    let ny=0.75*(3.0*x).sin()*(3.0*y).sin();
    let inv=(nx*nx+ny*ny+1.0).sqrt().recip();
    let o=object as f32;
    let texture=0.85+0.15*(4.0*x+2.0*y+o).sin();
    [x,y,z,nx*inv,ny*inv,inv,0.8944272*view.cos(),0.8944272*view.sin(),0.4472136,
     0.8944272*light.cos(),0.8944272*light.sin(),0.4472136,
     (0.18+0.10*o)*texture,(0.55-0.06*o)*texture,(0.25+0.04*o)*texture,
     if object==1 {0.1} else {0.35+0.10*o}, if object==1 {1.0} else {0.0}]
}
pub fn sample(width:usize,height:usize,pixel:usize,frame:u32,sequence:Sequence)->([f32;17],u32) {
    let u=((pixel%width) as f32+0.5)/width as f32*6.0;
    let object=(u.floor() as u32).min(5);
    let x=(u-object as f32)*2.0-1.0;
    let y=((pixel/width) as f32+0.5)/height as f32*2.0-1.0;
    let z=0.25*(3.0*x).sin()*(3.0*y).cos();
    let (v,l)=sequence.angles(frame);
    (features([x,y,z],object,v,l),object)
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn deterministic_scene(){assert_eq!(sample(128,72,57,99,Sequence::Camera),sample(128,72,57,99,Sequence::Camera));}
    #[test] fn trajectory(){assert_eq!(sample(128,72,57,0,Sequence::Static),sample(128,72,57,999,Sequence::Static));assert_ne!(sample(128,72,57,0,Sequence::Camera),sample(128,72,57,99,Sequence::Camera));}
}
