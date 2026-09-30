use serde::{Deserialize,Serialize};
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct ErrorMetrics {pub mse:f64,pub rmse:f64,pub psnr_db:Option<f64>,pub max_abs_error:f64}
impl ErrorMetrics {
    pub fn from_sums(sum_sq:f64,max_abs:f64,count:usize)->Self {
        let mse=sum_sq/count as f64;
        Self{mse,rmse:mse.sqrt(),psnr_db:if mse==0.0 {None}else{Some(-10.0*mse.log10())},max_abs_error:max_abs}
    }
    pub fn compare(a:&[f32],b:&[f32])->Self {
        assert_eq!(a.len(),b.len());assert!(!a.is_empty());
        let mut sum=0.0;let mut max:f64=0.0;
        for (&x,&y) in a.iter().zip(b){let d=(x as f64-y as f64).abs();sum+=d*d;max=max.max(d);}
        Self::from_sums(sum,max,a.len())
    }
}
pub fn quantile(v:&[f64],q:f64)->f64 {assert!(!v.is_empty());let mut s=v.to_vec();s.sort_by(f64::total_cmp);let x=q.clamp(0.0,1.0)*(s.len()-1) as f64;let a=x.floor() as usize;let b=x.ceil() as usize;s[a]+(s[b]-s[a])*(x-a as f64)}
#[cfg(test)]mod tests{use super::*;#[test]fn metrics(){let e=ErrorMetrics::compare(&[0.,0.,0.],&[1.,1.,1.]);assert_eq!(e.mse,1.);assert_eq!(e.psnr_db,Some(0.));assert_eq!(ErrorMetrics::compare(&[1.],&[1.]).psnr_db,None);assert_eq!(quantile(&[3.,1.,2.,4.],0.5),2.5);}}
