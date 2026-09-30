use crate::{config::CacheConfig,scene::features};
pub fn dense_index(c:&CacheConfig,o:usize,p:[usize;3],v:usize,l:usize)->Option<usize>{
    if o>=6||p.iter().zip(c.position_grid).any(|(&a,b)|a>=b)||v>=c.view_bins||l>=c.light_bins{return None;}
    Some(((((o*c.position_grid[0]+p[0])*c.position_grid[1]+p[1])*c.position_grid[2]+p[2])*c.view_bins+v)*c.light_bins+l)
}
pub fn quantize(c:&CacheConfig,object:usize,f:&[f32;17])->Option<usize>{
    if f[..3].iter().any(|v|!v.is_finite()||*v< -1.0||*v>1.0){return None;}
    let p=std::array::from_fn(|i|(((f[i]+1.0)*0.5*c.position_grid[i] as f32).floor() as usize).min(c.position_grid[i]-1));
    let bin=|y:f32,x:f32,n:usize|((y.atan2(x).rem_euclid(std::f32::consts::TAU)/std::f32::consts::TAU*n as f32).floor() as usize).min(n-1);
    dense_index(c,object,p,bin(f[7],f[6],c.view_bins),bin(f[10],f[9],c.light_bins))
}
pub fn representative(c:&CacheConfig,mut i:usize)->[f32;17]{
    let l=i%c.light_bins;i/=c.light_bins;let v=i%c.view_bins;i/=c.view_bins;
    let z=i%c.position_grid[2];i/=c.position_grid[2];let y=i%c.position_grid[1];i/=c.position_grid[1];let x=i%c.position_grid[0];let o=i/c.position_grid[0];
    let pos=[x,y,z].map(|a|a as f32);let p=std::array::from_fn(|j|(pos[j]+0.5)/c.position_grid[j] as f32*2.0-1.0);
    features(p,o as u32,(v as f32+0.5)/c.view_bins as f32*std::f32::consts::TAU,(l as f32+0.5)/c.light_bins as f32*std::f32::consts::TAU)
}
#[cfg(test)]mod tests{use super::*;#[test]fn bounds_quantization(){let c=CacheConfig{position_grid:[2,3,4],view_bins:4,light_bins:3,..Default::default()};for i in 0..c.entries().unwrap(){let f=representative(&c,i);let o=i/(2*3*4*4*3);assert_eq!(quantize(&c,o,&f),Some(i));}assert_eq!(dense_index(&c,6,[0;3],0,0),None);assert_eq!(dense_index(&c,0,[2,0,0],0,0),None);}}
/// CPU reference of `cache_lookup_interp`. Position axes clamp, azimuths wrap.
pub fn interpolate(c:&CacheConfig,object:usize,f:&[f32;17],cache:&[f32])->[f32;3]{
    let axis=|v:f32,g:usize|{let t=(v+1.0)*0.5*g as f32-0.5;let i0=t.floor().clamp(0.0,g as f32-1.0);let w=(t-i0).clamp(0.0,1.0);(i0 as usize,(i0 as usize+1).min(g-1),w)};
    let angle=|y:f32,x:f32,n:usize|{let t=y.atan2(x).rem_euclid(std::f32::consts::TAU)/std::f32::consts::TAU*n as f32-0.5;let f=t.floor();let i0=(f as i64).rem_euclid(n as i64) as usize;(i0,(i0+1)%n,t-f)};
    let a=[axis(f[0],c.position_grid[0]),axis(f[1],c.position_grid[1]),axis(f[2],c.position_grid[2]),angle(f[7],f[6],c.view_bins),angle(f[10],f[9],c.light_bins)];
    let mut rgb=[0.0f32;3];
    for corner in 0..32 {
        let pick=|d:usize|if corner>>d&1==1{(a[d].1,a[d].2)}else{(a[d].0,1.0-a[d].2)};
        let (x,wx)=pick(0);let (y,wy)=pick(1);let (z,wz)=pick(2);let (v,wv)=pick(3);let (l,wl)=pick(4);
        let i=dense_index(c,object,[x,y,z],v,l).expect("interpolation corner inside the dense cache");let w=wx*wy*wz*wv*wl;
        for (k,out) in rgb.iter_mut().enumerate(){*out+=w*cache[i*4+k];}
    }
    rgb
}
#[cfg(test)]mod interp_tests{use super::*;
    #[test]fn interpolation_reproduces_centres_and_affine_data(){
        let c=CacheConfig{position_grid:[3,4,2],view_bins:5,light_bins:3,..Default::default()};let n=c.entries().unwrap();
        let cache:Vec<f32>=(0..n).flat_map(|i|[i as f32,0.0,1.0,0.0]).collect();
        for i in (0..n).step_by(7){let f=representative(&c,i);let o=i/(n/6);let r=interpolate(&c,o,&f,&cache);assert!((r[0]-i as f32).abs()<1e-2,"centre {i}: {r:?}");assert!((r[2]-1.0).abs()<1e-5);}
        // Weights sum to one, clamped edges included.
        let f=crate::scene::features([0.99,-0.99,0.0],3,1.0,6.2);assert!((interpolate(&c,3,&f,&cache)[2]-1.0).abs()<1e-5);
    }
}
