use sha2::{Digest,Sha256};
pub const DIMS:[usize;6]=[17,128,128,128,64,3];
pub const PADDED:[usize;6]=[32,128,128,128,64,32];
#[derive(Clone)] pub struct Teacher { pub weights:Vec<Vec<f32>>, pub biases:Vec<Vec<f32>> }
struct Rng(u64);
impl Rng {
    fn unit(&mut self)->f32 { self.0=self.0.wrapping_add(0x9e3779b97f4a7c15);let mut z=self.0;z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb);((z^(z>>31))>>40) as f32/16777216.0 }
}
pub fn gelu(x:f32)->f32 {0.5*x*(1.0+(0.7978846*(x+0.044715*x*x*x)).tanh())}
impl Teacher {
    pub fn new(seed:u64)->Self {
        let mut rng=Rng(seed);let mut weights=Vec::new();let mut biases=Vec::new();
        for l in 0..5 {
            let mut w=vec![0.0;PADDED[l]*PADDED[l+1]];
            let scale=1.65*(3.0/DIMS[l] as f32).sqrt();
            for i in 0..DIMS[l] {for j in 0..DIMS[l+1] {w[i*PADDED[l+1]+j]=(rng.unit()*2.0-1.0)*scale;}}
            let mut b=vec![0.0;PADDED[l+1]];
            for v in b.iter_mut().take(DIMS[l+1]){*v=(rng.unit()*2.0-1.0)*0.08;}
            weights.push(w);biases.push(b);
        }Self{weights,biases}
    }
    pub fn forward(&self,input:&[f32;17])->[f32;3] {
        let mut a=input.to_vec();
        for l in 0..5 {let mut b=vec![0.0;DIMS[l+1]];for (j,v) in b.iter_mut().enumerate(){*v=self.biases[l][j];for (i,x) in a.iter().enumerate(){*v+=x*self.weights[l][i*PADDED[l+1]+j];}*v=if l==4 {1.0/(1.0+(-*v).exp())} else {gelu(*v)};}a=b;}
        [a[0],a[1],a[2]]
    }
    pub fn hash(&self)->String {let mut h=Sha256::new();for v in self.weights.iter().chain(self.biases.iter()).flatten(){h.update(v.to_le_bytes());}format!("{:x}",h.finalize())}
}
#[cfg(test)] mod tests {use super::*;#[test]fn teacher_repeat(){let t=Teacher::new(1337);let a=[0.2;17];assert_eq!(t.forward(&a),Teacher::new(1337).forward(&a));assert_ne!(t.forward(&a),t.forward(&[0.8;17]));assert_ne!(t.hash(),Teacher::new(7).hash());}}
