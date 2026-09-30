#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn reduce_mse(out:&mut Tensor<f32,{[1,2]}>,errors:&Tensor<f32,{[-1,2]}>,blocks:i32) {
        let p=errors.partition(shape![256,1]);
        let mut sum:Tile<f32,{[]}>=constant(0.0f32,shape![]);
        let mut max:Tile<f32,{[]}>=constant(0.0f32,shape![]);
        for i in 0i32..blocks {
            let a=p.load([i,0i32]).reshape(shape![256]);
            let b=p.load([i,1i32]).reshape(shape![256]);
            let s:Tile<f32,{[]}>=reduce_sum(a,0i32);let m:Tile<f32,{[]}>=reduce_max(b,0i32);
            sum=sum+s;max=max_tile(max,m);
        }
        let cols=iota::<i32,{[2]}>(shape![2]).reshape(shape![1,2]);let zero:Tile<i32,{[1,2]}>=constant(0i32,shape![1,2]);
        let v=select(eq_tile(cols,zero),sum.reshape(shape![1,1]).broadcast(shape![1,2]),max.reshape(shape![1,1]).broadcast(shape![1,2]));out.store(v);
    }

    #[cutile::entry()]
    pub fn pack_rgb(out:&mut Tensor<f32,{[32,4]}>,rgb:&Tensor<f32,{[-1,32]}>) {
        let pid=get_tile_block_id();
        let v=rgb.partition(shape![32,4]).load([pid.0,0i32]);
        out.store(v);
    }
    #[cutile::entry()]
    pub fn quantize_cache_key(out:&mut Tensor<i32,{[32]}>,features:&Tensor<f32,{[-1,32]}>,width:i32,gx:i32,gy:i32,gz:i32,vbins:i32,lbins:i32) {
        let pid=get_tile_block_id();let fp=features.partition(shape![32,1]);
        let one:Tile<f32,{[32]}>=constant(1.0f32,shape![32]);
        let half:Tile<f32,{[32]}>=constant(0.5f32,shape![32]);
        let zero:Tile<f32,{[32]}>=constant(0.0f32,shape![32]);
        let tau:Tile<f32,{[32]}>=constant(6.283185307f32,shape![32]);
        let xi:Tile<f32,{[32]}>=fp.load([pid.0,0i32]).reshape(shape![32]);
        let yi:Tile<f32,{[32]}>=fp.load([pid.0,1i32]).reshape(shape![32]);
        let zi:Tile<f32,{[32]}>=fp.load([pid.0,2i32]).reshape(shape![32]);
        let gxf:Tile<f32,{[32]}>=broadcast_scalar(convert_scalar::<f32>(gx),shape![32]);
        let gyf:Tile<f32,{[32]}>=broadcast_scalar(convert_scalar::<f32>(gy),shape![32]);
        let gzf:Tile<f32,{[32]}>=broadcast_scalar(convert_scalar::<f32>(gz),shape![32]);
        let x:Tile<i32,{[32]}>=convert_tile(min_tile(max_tile(floor((xi+one)*half*gxf),zero),gxf-one));
        let y:Tile<i32,{[32]}>=convert_tile(min_tile(max_tile(floor((yi+one)*half*gyf),zero),gyf-one));
        let z:Tile<i32,{[32]}>=convert_tile(min_tile(max_tile(floor((zi+one)*half*gzf),zero),gzf-one));
        let vx=fp.load([pid.0,6i32]).reshape(shape![32]);let vy=fp.load([pid.0,7i32]).reshape(shape![32]);
        let lx=fp.load([pid.0,9i32]).reshape(shape![32]);let ly=fp.load([pid.0,10i32]).reshape(shape![32]);
        let va=atan2(vy,vx);let la=atan2(ly,lx);
        let vp=select(lt_tile(va,zero),va+tau,va);let lp=select(lt_tile(la,zero),la+tau,la);
        let vf:Tile<f32,{[32]}>=broadcast_scalar(convert_scalar::<f32>(vbins),shape![32]);
        let lf:Tile<f32,{[32]}>=broadcast_scalar(convert_scalar::<f32>(lbins),shape![32]);
        let v:Tile<i32,{[32]}>=convert_tile(min_tile(floor(vp/tau*vf),vf-one));
        let l:Tile<i32,{[32]}>=convert_tile(min_tile(floor(lp/tau*lf),lf-one));
        let ids:Tile<i32,{[32]}>=iota(shape![32])+broadcast_scalar(pid.0*32,shape![32]);
        let px:Tile<f32,{[32]}>=convert_tile(ids%broadcast_scalar(width,shape![32]));
        let ob:Tile<i32,{[32]}>=convert_tile(floor((px+half)*broadcast_scalar(6.0f32/convert_scalar::<f32>(width),shape![32])));
        let key=ob*broadcast_scalar(gx,shape![32])+x;
        let key=key*broadcast_scalar(gy,shape![32])+y;
        let key=key*broadcast_scalar(gz,shape![32])+z;
        let key=key*broadcast_scalar(vbins,shape![32])+v;
        let key=key*broadcast_scalar(lbins,shape![32])+l;
        out.store(key);
    }
    #[cutile::entry()]
    pub fn cache_lookup_nearest(out:&mut Tensor<f32,{[32,4]}>,keys:&Tensor<i32,{[-1]}>,cache:&Tensor<f32,{[-1,4]}>,fallback:&Tensor<f32,{[-1,32]}>,per_object:i32,invalid_mask:i32) {
        let pid=get_tile_block_id();
        let key=keys.partition(shape![32]).load([pid.0]);
        let four:Tile<i32,{[32]}>=constant(4i32,shape![32]);
        let base=(key*four).reshape(shape![32,1]).broadcast(shape![32,4]);
        let channels=iota::<i32,{[4]}>(shape![4]).reshape(shape![1,4]).broadcast(shape![32,4]);
        let address:PointerTile<*const f32,{[]}>=pointer_to_tile(cache.as_ptr());
        let address2:PointerTile<*const f32,{[1,1]}>=address.reshape(shape![1,1]);
        let addresses:PointerTile<*const f32,{[32,4]}>=address2.broadcast(shape![32,4]);
        let ptr:PointerTile<*const f32,{[32,4]}>=addresses.offset_tile(base+channels);
        // Keys are clamped in range by quantize_cache_key.
        let loaded:(Tile<f32,{[32,4]}>,Token)=unsafe{load_ptr_tko(ptr,ordering::Weak,None::<scope::TileBlock>,None,None::<f32>,None,Latency::<0>)};
        let one:Tile<i32,{[32]}>=constant(1i32,shape![32]);let zero:Tile<i32,{[32]}>=constant(0i32,shape![32]);
        let obj=key/broadcast_scalar(per_object,shape![32]);
        let invalid=broadcast_scalar(invalid_mask,shape![32])&(one<<obj);
        let valid=eq_tile(invalid,zero).reshape(shape![32,1]).broadcast(shape![32,4]);
        let mut rgb=loaded.0;
        if invalid_mask!=0 {let full=fallback.partition(shape![32,4]).load([pid.0,0i32]);rgb=select(valid,rgb,full);}
        let three:Tile<i32,{[32,4]}>=constant(3i32,shape![32,4]);
        let yes:Tile<f32,{[32,4]}>=constant(1.0f32,shape![32,4]);let no:Tile<f32,{[32,4]}>=constant(0.0f32,shape![32,4]);
        let result=select(eq_tile(channels,three),select(valid,yes,no),rgb);
        out.store(result);
    }
    /// Multilinear lookup over x, y, z, view and light (32 corners).
    /// Position axes clamp, azimuths wrap.
    #[cutile::entry()]
    pub fn cache_lookup_interp(out:&mut Tensor<f32,{[32,4]}>,features:&Tensor<f32,{[-1,32]}>,cache:&Tensor<f32,{[-1,4]}>,width:i32,gx:i32,gy:i32,gz:i32,vbins:i32,lbins:i32) {
        let pid=get_tile_block_id();let fp=features.partition(shape![32,1]);
        let one:Tile<f32,{[32]}>=constant(1.0f32,shape![32]);
        let half:Tile<f32,{[32]}>=constant(0.5f32,shape![32]);
        let zero:Tile<f32,{[32]}>=constant(0.0f32,shape![32]);
        let tau:Tile<f32,{[32]}>=constant(6.283185307f32,shape![32]);
        let onei:Tile<i32,{[32]}>=constant(1i32,shape![32]);
        let twoi:Tile<i32,{[32]}>=constant(2i32,shape![32]);
        let gxf:Tile<f32,{[32]}>=broadcast_scalar(convert_scalar::<f32>(gx),shape![32]);
        let gyf:Tile<f32,{[32]}>=broadcast_scalar(convert_scalar::<f32>(gy),shape![32]);
        let gzf:Tile<f32,{[32]}>=broadcast_scalar(convert_scalar::<f32>(gz),shape![32]);
        let vf:Tile<f32,{[32]}>=broadcast_scalar(convert_scalar::<f32>(vbins),shape![32]);
        let lf:Tile<f32,{[32]}>=broadcast_scalar(convert_scalar::<f32>(lbins),shape![32]);
        let gxi:Tile<i32,{[32]}>=broadcast_scalar(gx,shape![32]);
        let gyi:Tile<i32,{[32]}>=broadcast_scalar(gy,shape![32]);
        let gzi:Tile<i32,{[32]}>=broadcast_scalar(gz,shape![32]);
        let vbi:Tile<i32,{[32]}>=broadcast_scalar(vbins,shape![32]);
        let lbi:Tile<i32,{[32]}>=broadcast_scalar(lbins,shape![32]);
        let xs:Tile<f32,{[32]}>=fp.load([pid.0,0i32]).reshape(shape![32]);
        let ys:Tile<f32,{[32]}>=fp.load([pid.0,1i32]).reshape(shape![32]);
        let zs:Tile<f32,{[32]}>=fp.load([pid.0,2i32]).reshape(shape![32]);
        let tx=(xs+one)*half*gxf-half;let x0f=min_tile(max_tile(floor(tx),zero),gxf-one);let wx=min_tile(max_tile(tx-x0f,zero),one);
        let ty=(ys+one)*half*gyf-half;let y0f=min_tile(max_tile(floor(ty),zero),gyf-one);let wy=min_tile(max_tile(ty-y0f,zero),one);
        let tz=(zs+one)*half*gzf-half;let z0f=min_tile(max_tile(floor(tz),zero),gzf-one);let wz=min_tile(max_tile(tz-z0f,zero),one);
        let x0:Tile<i32,{[32]}>=convert_tile(x0f);let x1:Tile<i32,{[32]}>=convert_tile(min_tile(x0f+one,gxf-one));
        let y0:Tile<i32,{[32]}>=convert_tile(y0f);let y1:Tile<i32,{[32]}>=convert_tile(min_tile(y0f+one,gyf-one));
        let z0:Tile<i32,{[32]}>=convert_tile(z0f);let z1:Tile<i32,{[32]}>=convert_tile(min_tile(z0f+one,gzf-one));
        let vx=fp.load([pid.0,6i32]).reshape(shape![32]);let vy=fp.load([pid.0,7i32]).reshape(shape![32]);
        let lx=fp.load([pid.0,9i32]).reshape(shape![32]);let ly=fp.load([pid.0,10i32]).reshape(shape![32]);
        let va=atan2(vy,vx);let la=atan2(ly,lx);
        let vp=select(lt_tile(va,zero),va+tau,va);let lp=select(lt_tile(la,zero),la+tau,la);
        let tv=vp/tau*vf-half;let fv=floor(tv);let wv=tv-fv;
        let tl=lp/tau*lf-half;let fl=floor(tl);let wl=tl-fl;
        let v0i:Tile<i32,{[32]}>=convert_tile(fv);let l0i:Tile<i32,{[32]}>=convert_tile(fl);
        let v0=(v0i+vbi)%vbi;let v1=(v0+onei)%vbi;
        let l0=(l0i+lbi)%lbi;let l1=(l0+onei)%lbi;
        let ids:Tile<i32,{[32]}>=iota(shape![32])+broadcast_scalar(pid.0*32,shape![32]);
        let px:Tile<f32,{[32]}>=convert_tile(ids%broadcast_scalar(width,shape![32]));
        let ob:Tile<i32,{[32]}>=convert_tile(floor((px+half)*broadcast_scalar(6.0f32/convert_scalar::<f32>(width),shape![32])));
        let four:Tile<i32,{[32]}>=constant(4i32,shape![32]);
        let channels=iota::<i32,{[4]}>(shape![4]).reshape(shape![1,4]).broadcast(shape![32,4]);
        let address:PointerTile<*const f32,{[]}>=pointer_to_tile(cache.as_ptr());
        let address2:PointerTile<*const f32,{[1,1]}>=address.reshape(shape![1,1]);
        let addresses:PointerTile<*const f32,{[32,4]}>=address2.broadcast(shape![32,4]);
        let mut acc:Tile<f32,{[32,4]}>=constant(0.0f32,shape![32,4]);
        for corner in 0i32..32i32 {
            let cb:Tile<i32,{[32]}>=broadcast_scalar(corner,shape![32]);
            let bx=cb%twoi;let by=(cb/twoi)%twoi;let bz=(cb/twoi/twoi)%twoi;let bv=(cb/twoi/twoi/twoi)%twoi;let bl=cb/twoi/twoi/twoi/twoi;
            let bxf:Tile<f32,{[32]}>=convert_tile(bx);let byf:Tile<f32,{[32]}>=convert_tile(by);let bzf:Tile<f32,{[32]}>=convert_tile(bz);
            let bvf:Tile<f32,{[32]}>=convert_tile(bv);let blf:Tile<f32,{[32]}>=convert_tile(bl);
            let w=(bxf*wx+(one-bxf)*(one-wx))*(byf*wy+(one-byf)*(one-wy))*(bzf*wz+(one-bzf)*(one-wz))*(bvf*wv+(one-bvf)*(one-wv))*(blf*wl+(one-blf)*(one-wl));
            let key=ob*gxi+x0+bx*(x1-x0);
            let key=key*gyi+y0+by*(y1-y0);
            let key=key*gzi+z0+bz*(z1-z0);
            let key=key*vbi+v0+bv*(v1-v0);
            let key=key*lbi+l0+bl*(l1-l0);
            let base=(key*four).reshape(shape![32,1]).broadcast(shape![32,4]);
            let ptr:PointerTile<*const f32,{[32,4]}>=addresses.offset_tile(base+channels);
            // Corner indices are clamped or wrapped in range.
            let loaded:(Tile<f32,{[32,4]}>,Token)=unsafe{load_ptr_tko(ptr,ordering::Weak,None::<scope::TileBlock>,None,None::<f32>,None,Latency::<0>)};
            acc=acc+loaded.0*w.reshape(shape![32,1]).broadcast(shape![32,4]);
        }
        let three:Tile<i32,{[32,4]}>=constant(3i32,shape![32,4]);
        let yes:Tile<f32,{[32,4]}>=constant(1.0f32,shape![32,4]);
        out.store(select(eq_tile(channels,three),yes,acc));
    }
    /// Column 0: keys unchanged since the previous frame, per 32-sample block.
    #[cutile::entry()]
    pub fn key_match(out:&mut Tensor<f32,{[1,2]}>,keys:&Tensor<i32,{[-1]}>,prev:&Tensor<i32,{[-1]}>) {
        let pid=get_tile_block_id();
        let a=keys.partition(shape![32]).load([pid.0]);let b=prev.partition(shape![32]).load([pid.0]);
        let one:Tile<f32,{[32]}>=constant(1.0f32,shape![32]);let zero:Tile<f32,{[32]}>=constant(0.0f32,shape![32]);
        let same:Tile<f32,{[]}>=reduce_sum(select(eq_tile(a,b),one,zero),0i32);
        let cols=iota::<i32,{[2]}>(shape![2]).reshape(shape![1,2]);let zi:Tile<i32,{[1,2]}>=constant(0i32,shape![1,2]);let zf:Tile<f32,{[1,2]}>=constant(0.0f32,shape![1,2]);
        out.store(select(eq_tile(cols,zi),same.reshape(shape![1,1]).broadcast(shape![1,2]),zf));
    }
    #[cutile::entry()]
    pub fn error_rgb(out:&mut Tensor<f32,{[1,2]}>,a:&Tensor<f32,{[-1,32]}>,b:&Tensor<f32,{[-1,4]}>) {
        let pid=get_tile_block_id();
        let aa=a.partition(shape![32,4]).load([pid.0,0i32]);let bb=b.partition(shape![32,4]).load([pid.0,0i32]);
        let channels=iota::<i32,{[4]}>(shape![4]).reshape(shape![1,4]).broadcast(shape![32,4]);
        let three:Tile<i32,{[32,4]}>=constant(3i32,shape![32,4]);let zero:Tile<f32,{[32,4]}>=constant(0.0f32,shape![32,4]);
        let d=select(lt_tile(channels,three),aa-bb,zero);
        let row_sum:Tile<f32,{[32]}>=reduce_sum(d*d,1i32);
        let squared:Tile<f32,{[]}>=reduce_sum(row_sum,0i32);
        let absolute=max_tile(d,zero-d);
        let row_max:Tile<f32,{[32]}>=reduce_max(absolute,1i32);
        let mx:Tile<f32,{[]}>=reduce_max(row_max,0i32);
        let cols=iota::<i32,{[2]}>(shape![2]).reshape(shape![1,2]);let zi:Tile<i32,{[1,2]}>=constant(0i32,shape![1,2]);
        let values=select(eq_tile(cols,zi),squared.reshape(shape![1,1]).broadcast(shape![1,2]),mx.reshape(shape![1,1]).broadcast(shape![1,2]));out.store(values);
    }

    #[cutile::entry()]
    pub fn set_frame(out:&mut Tensor<i32,{[1]}>, value:i32) {
        let v:Tile<i32,{[1]}>=broadcast_scalar(value,shape![1]);
        out.store(v);
    }
    #[cutile::entry()]
    pub fn advance(out:&mut Tensor<i32,{[1]}>) {
        let ct1:Tile<i32,{[1]}>=constant(1i32,shape![1]);
        let value=out.load()+ct1;
        out.store(value);
    }
    #[cutile::entry()]
    pub fn generate_frame_features(out:&mut Tensor<f32,{[32,32]}>, frame:&Tensor<i32,{[1]}>, width:i32,height:i32,sequence:i32) {
        let ct2:Tile<f32,{[32]}>=constant(0.5f32,shape![32]);
        let ct3:Tile<f32,{[32]}>=constant(2.0f32,shape![32]);
        let ct4:Tile<f32,{[32]}>=constant(1.0f32,shape![32]);
        let ct5:Tile<f32,{[32]}>=constant(0.5f32,shape![32]);
        let ct6:Tile<f32,{[32]}>=constant(1.0f32,shape![32]);
        let ct7:Tile<f32,{[32]}>=constant(0.25f32,shape![32]);
        let ct8:Tile<f32,{[32]}>=constant(3.0f32,shape![32]);
        let ct10:Tile<f32,{[32]}>=constant(0.0f32,shape![32]);
        let ct11:Tile<f32,{[32]}>=constant(0.0f32,shape![32]);
        let ct12:Tile<f32,{[32]}>=constant(0.013f32,shape![32]);
        let ct13:Tile<f32,{[32]}>=constant(0.0031f32,shape![32]);
        let ct14:Tile<f32,{[32]}>=constant(-0.75f32,shape![32]);
        let ct15:Tile<f32,{[32]}>=constant(3.0f32,shape![32]);
        let ct17:Tile<f32,{[32]}>=constant(0.75f32,shape![32]);
        let ct18:Tile<f32,{[32]}>=constant(3.0f32,shape![32]);
        let ct20:Tile<f32,{[32]}>=constant(1.0f32,shape![32]);
        let ct22:Tile<f32,{[32]}>=constant(0.85f32,shape![32]);
        let ct23:Tile<f32,{[32]}>=constant(0.15f32,shape![32]);
        let ct24:Tile<f32,{[32]}>=constant(4.0f32,shape![32]);
        let ct25:Tile<f32,{[32]}>=constant(2.0f32,shape![32]);
        let ct26:Tile<f32,{[32,32]}>=constant(0.0f32,shape![32,32]);
        let ct27:Tile<i32,{[32,32]}>=constant(0i32,shape![32,32]);
        let ct28:Tile<i32,{[32,32]}>=constant(1i32,shape![32,32]);
        let ct29:Tile<i32,{[32,32]}>=constant(2i32,shape![32,32]);
        let ct30:Tile<i32,{[32,32]}>=constant(3i32,shape![32,32]);
        let ct31:Tile<i32,{[32,32]}>=constant(4i32,shape![32,32]);
        let ct32:Tile<i32,{[32,32]}>=constant(5i32,shape![32,32]);
        let ct33:Tile<i32,{[32,32]}>=constant(6i32,shape![32,32]);
        let ct34:Tile<f32,{[32]}>=constant(0.8944272f32,shape![32]);
        let ct35:Tile<i32,{[32,32]}>=constant(7i32,shape![32,32]);
        let ct36:Tile<f32,{[32]}>=constant(0.8944272f32,shape![32]);
        let ct37:Tile<i32,{[32,32]}>=constant(8i32,shape![32,32]);
        let ct38:Tile<f32,{[32,32]}>=constant(0.4472136f32,shape![32,32]);
        let ct39:Tile<i32,{[32,32]}>=constant(9i32,shape![32,32]);
        let ct40:Tile<f32,{[32]}>=constant(0.8944272f32,shape![32]);
        let ct41:Tile<i32,{[32,32]}>=constant(10i32,shape![32,32]);
        let ct42:Tile<f32,{[32]}>=constant(0.8944272f32,shape![32]);
        let ct43:Tile<i32,{[32,32]}>=constant(11i32,shape![32,32]);
        let ct44:Tile<f32,{[32,32]}>=constant(0.4472136f32,shape![32,32]);
        let ct45:Tile<i32,{[32,32]}>=constant(12i32,shape![32,32]);
        let ct46:Tile<f32,{[32]}>=constant(0.18f32,shape![32]);
        let ct47:Tile<f32,{[32]}>=constant(0.10f32,shape![32]);
        let ct48:Tile<i32,{[32,32]}>=constant(13i32,shape![32,32]);
        let ct49:Tile<f32,{[32]}>=constant(0.55f32,shape![32]);
        let ct50:Tile<f32,{[32]}>=constant(0.06f32,shape![32]);
        let ct51:Tile<i32,{[32,32]}>=constant(14i32,shape![32,32]);
        let ct52:Tile<f32,{[32]}>=constant(0.25f32,shape![32]);
        let ct53:Tile<f32,{[32]}>=constant(0.04f32,shape![32]);
        let ct54:Tile<f32,{[32]}>=constant(1.0f32,shape![32]);
        let ct55:Tile<f32,{[32]}>=constant(0.1f32,shape![32]);
        let ct56:Tile<f32,{[32]}>=constant(0.35f32,shape![32]);
        let ct57:Tile<f32,{[32]}>=constant(0.10f32,shape![32]);
        let ct58:Tile<i32,{[32,32]}>=constant(15i32,shape![32,32]);
        let ct59:Tile<i32,{[32,32]}>=constant(16i32,shape![32,32]);
        let ct60:Tile<f32,{[32]}>=constant(1.0f32,shape![32]);
        let ct61:Tile<f32,{[32]}>=constant(0.0f32,shape![32]);
        let pid=get_tile_block_id();
        let idx=iota::<i32,{[32]}>(shape![32])+broadcast_scalar(pid.0*32,shape![32]);
        let px:Tile<f32,{[32]}>=convert_tile(idx%broadcast_scalar(width,shape![32]));
        let py:Tile<f32,{[32]}>=convert_tile(idx/broadcast_scalar(width,shape![32]));
        let u=(px+ct2)*broadcast_scalar(6.0f32/convert_scalar::<f32>(width),shape![32]);
        let o=floor(u);
        let x=(u-o)*ct3-ct4;
        let y=(py+ct5)*broadcast_scalar(2.0f32/convert_scalar::<f32>(height),shape![32])-ct6;
        let z=ct7*sin(x*ct8)*cos(y*ct8);
        let f:Tile<f32,{[1]}>=convert_tile(frame.load_tile(shape![1],[0i32]));
        let ft=f.broadcast(shape![32]);
        let mut va=ct10;
        let mut la=ct11;
        if sequence>0 {va=ft*ct12;}
        if sequence>1 {la=ft*ct13;}
        let nx=ct14*cos(x*ct15)*cos(y*ct15);
        let ny=ct17*sin(x*ct18)*sin(y*ct18);
        let inv=ct20/sqrt(nx*nx+ny*ny+ct20,rounding::NearestEven,ftz::Disabled);
        let texture=ct22+ct23*sin(x*ct24+y*ct25+o);
        let c=iota::<i32,{[32]}>(shape![32]).reshape(shape![1,32]).broadcast(shape![32,32]);
        let mut v=ct26;
        v=select(eq_tile(c,ct27),x.reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct28),y.reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct29),z.reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct30),(nx*inv).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct31),(ny*inv).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct32),inv.reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct33),(cos(va)*ct34).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct35),(sin(va)*ct36).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct37),ct38,v);
        v=select(eq_tile(c,ct39),(cos(la)*ct40).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct41),(sin(la)*ct42).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct43),ct44,v);
        v=select(eq_tile(c,ct45),((ct46+o*ct47)*texture).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct48),((ct49-o*ct50)*texture).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct51),((ct52+o*ct53)*texture).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        let metal=eq_tile(o,ct54);
        let rough=select(metal,ct55,ct56+o*ct57);
        v=select(eq_tile(c,ct58),rough.reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct59),select(metal,ct60,ct61).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        out.store(v);
    }
    #[cutile::entry()]
    pub fn generate_bake_features(out:&mut Tensor<f32,{[32,32]}>, start:i32,gx:i32,gy:i32,gz:i32,vbins:i32,lbins:i32) {
        let ct2:Tile<f32,{[32]}>=constant(0.5f32,shape![32]);
        let ct4:Tile<f32,{[32]}>=constant(1.0f32,shape![32]);
        let ct14:Tile<f32,{[32]}>=constant(-0.75f32,shape![32]);
        let ct15:Tile<f32,{[32]}>=constant(3.0f32,shape![32]);
        let ct17:Tile<f32,{[32]}>=constant(0.75f32,shape![32]);
        let ct18:Tile<f32,{[32]}>=constant(3.0f32,shape![32]);
        let ct20:Tile<f32,{[32]}>=constant(1.0f32,shape![32]);
        let ct22:Tile<f32,{[32]}>=constant(0.85f32,shape![32]);
        let ct23:Tile<f32,{[32]}>=constant(0.15f32,shape![32]);
        let ct24:Tile<f32,{[32]}>=constant(4.0f32,shape![32]);
        let ct25:Tile<f32,{[32]}>=constant(2.0f32,shape![32]);
        let ct26:Tile<f32,{[32,32]}>=constant(0.0f32,shape![32,32]);
        let ct27:Tile<i32,{[32,32]}>=constant(0i32,shape![32,32]);
        let ct28:Tile<i32,{[32,32]}>=constant(1i32,shape![32,32]);
        let ct29:Tile<i32,{[32,32]}>=constant(2i32,shape![32,32]);
        let ct30:Tile<i32,{[32,32]}>=constant(3i32,shape![32,32]);
        let ct31:Tile<i32,{[32,32]}>=constant(4i32,shape![32,32]);
        let ct32:Tile<i32,{[32,32]}>=constant(5i32,shape![32,32]);
        let ct33:Tile<i32,{[32,32]}>=constant(6i32,shape![32,32]);
        let ct34:Tile<f32,{[32]}>=constant(0.8944272f32,shape![32]);
        let ct35:Tile<i32,{[32,32]}>=constant(7i32,shape![32,32]);
        let ct36:Tile<f32,{[32]}>=constant(0.8944272f32,shape![32]);
        let ct37:Tile<i32,{[32,32]}>=constant(8i32,shape![32,32]);
        let ct38:Tile<f32,{[32,32]}>=constant(0.4472136f32,shape![32,32]);
        let ct39:Tile<i32,{[32,32]}>=constant(9i32,shape![32,32]);
        let ct40:Tile<f32,{[32]}>=constant(0.8944272f32,shape![32]);
        let ct41:Tile<i32,{[32,32]}>=constant(10i32,shape![32,32]);
        let ct42:Tile<f32,{[32]}>=constant(0.8944272f32,shape![32]);
        let ct43:Tile<i32,{[32,32]}>=constant(11i32,shape![32,32]);
        let ct44:Tile<f32,{[32,32]}>=constant(0.4472136f32,shape![32,32]);
        let ct45:Tile<i32,{[32,32]}>=constant(12i32,shape![32,32]);
        let ct46:Tile<f32,{[32]}>=constant(0.18f32,shape![32]);
        let ct47:Tile<f32,{[32]}>=constant(0.10f32,shape![32]);
        let ct48:Tile<i32,{[32,32]}>=constant(13i32,shape![32,32]);
        let ct49:Tile<f32,{[32]}>=constant(0.55f32,shape![32]);
        let ct50:Tile<f32,{[32]}>=constant(0.06f32,shape![32]);
        let ct51:Tile<i32,{[32,32]}>=constant(14i32,shape![32,32]);
        let ct52:Tile<f32,{[32]}>=constant(0.25f32,shape![32]);
        let ct53:Tile<f32,{[32]}>=constant(0.04f32,shape![32]);
        let ct54:Tile<f32,{[32]}>=constant(1.0f32,shape![32]);
        let ct55:Tile<f32,{[32]}>=constant(0.1f32,shape![32]);
        let ct56:Tile<f32,{[32]}>=constant(0.35f32,shape![32]);
        let ct57:Tile<f32,{[32]}>=constant(0.10f32,shape![32]);
        let ct58:Tile<i32,{[32,32]}>=constant(15i32,shape![32,32]);
        let ct59:Tile<i32,{[32,32]}>=constant(16i32,shape![32,32]);
        let ct60:Tile<f32,{[32]}>=constant(1.0f32,shape![32]);
        let ct61:Tile<f32,{[32]}>=constant(0.0f32,shape![32]);
        let pid=get_tile_block_id();
        let idx:Tile<i32,{[32]}>=iota(shape![32])+broadcast_scalar(start+pid.0*32,shape![32]);
        let tx:Tile<i32,{[32]}>=broadcast_scalar(gx,shape![32]);
        let ty:Tile<i32,{[32]}>=broadcast_scalar(gy,shape![32]);
        let tz:Tile<i32,{[32]}>=broadcast_scalar(gz,shape![32]);
        let tv:Tile<i32,{[32]}>=broadcast_scalar(vbins,shape![32]);
        let tl:Tile<i32,{[32]}>=broadcast_scalar(lbins,shape![32]);
        let li:Tile<f32,{[32]}>=convert_tile(idx%tl);
        let vi:Tile<f32,{[32]}>=convert_tile((idx/tl)%tv);
        let zi:Tile<f32,{[32]}>=convert_tile((idx/tl/tv)%tz);
        let yi:Tile<f32,{[32]}>=convert_tile((idx/tl/tv/tz)%ty);
        let xi:Tile<f32,{[32]}>=convert_tile((idx/tl/tv/tz/ty)%tx);
        let o:Tile<f32,{[32]}>=convert_tile(idx/tl/tv/tz/ty/tx);
        let x=(xi+ct2)*broadcast_scalar(2.0f32/convert_scalar::<f32>(gx),shape![32])-ct4;
        let y=(yi+ct2)*broadcast_scalar(2.0f32/convert_scalar::<f32>(gy),shape![32])-ct4;
        let z=(zi+ct2)*broadcast_scalar(2.0f32/convert_scalar::<f32>(gz),shape![32])-ct4;
        let va=(vi+ct2)*broadcast_scalar(6.283185307f32/convert_scalar::<f32>(vbins),shape![32]);
        let la=(li+ct2)*broadcast_scalar(6.283185307f32/convert_scalar::<f32>(lbins),shape![32]);
        let nx=ct14*cos(x*ct15)*cos(y*ct15);
        let ny=ct17*sin(x*ct18)*sin(y*ct18);
        let inv=ct20/sqrt(nx*nx+ny*ny+ct20,rounding::NearestEven,ftz::Disabled);
        let texture=ct22+ct23*sin(x*ct24+y*ct25+o);
        let c=iota::<i32,{[32]}>(shape![32]).reshape(shape![1,32]).broadcast(shape![32,32]);
        let mut v=ct26;
        v=select(eq_tile(c,ct27),x.reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct28),y.reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct29),z.reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct30),(nx*inv).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct31),(ny*inv).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct32),inv.reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct33),(cos(va)*ct34).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct35),(sin(va)*ct36).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct37),ct38,v);
        v=select(eq_tile(c,ct39),(cos(la)*ct40).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct41),(sin(la)*ct42).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct43),ct44,v);
        v=select(eq_tile(c,ct45),((ct46+o*ct47)*texture).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct48),((ct49-o*ct50)*texture).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct51),((ct52+o*ct53)*texture).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        let metal=eq_tile(o,ct54);
        let rough=select(metal,ct55,ct56+o*ct57);
        v=select(eq_tile(c,ct58),rough.reshape(shape![32,1]).broadcast(shape![32,32]),v);
        v=select(eq_tile(c,ct59),select(metal,ct60,ct61).reshape(shape![32,1]).broadcast(shape![32,32]),v);
        out.store(v);
    }
    #[cutile::entry()]
    pub fn teacher_layer<const K:i32,const LAST:i32,const BM:i32,const BN:i32>(out:&mut Tensor<f32,{[BM,BN]}>,x:&Tensor<f32,{[-1,K]}>,w:&Tensor<f32,{[K,-1]}>,bias:&Tensor<f32,{[-1]}>) {
        let ct62:Tile<f32,{[BM,BN]}>=constant(0.0f32,shape![BM,BN]);
        let ct63:Tile<f32,{[BM,BN]}>=constant(1.0f32,shape![BM,BN]);
        let ct64:Tile<f32,{[BM,BN]}>=constant(0.0f32,shape![BM,BN]);
        let ct65:Tile<f32,{[BM,BN]}>=constant(0.5f32,shape![BM,BN]);
        let ct66:Tile<f32,{[BM,BN]}>=constant(0.7978846f32,shape![BM,BN]);
        let ct67:Tile<f32,{[BM,BN]}>=constant(0.044715f32,shape![BM,BN]);
        let pid=get_tile_block_id();
        let xp=x.partition(shape![BM,32]);
        let wp=w.partition(shape![32,BN]);
        let mut acc=ct62;
        for k in 0i32..(K/32) {acc=mma(xp.load([pid.0,k]),wp.load([k,pid.1]),acc);}
        let b=bias.partition(shape![BN]).load([pid.1]).reshape(shape![1,BN]).broadcast(shape![BM,BN]);
        let a=acc+b;
        let one=ct63;
        let mut result=ct65*a*(one+tanh(ct66*(a+ct67*a*a*a)));
        if LAST==1 { result=one/(one+exp(ct64-a)); }
        out.store(result);
    }
}
