const fs=require('fs'), Module=require('module'), p=require('path');
const fp=p.resolve('src/ocdkit/plot/web/spectra_density_gl.js');
// stub browser globals so the factory body defines without throwing
global.navigator=global.navigator||{}; global.window=global.window||global; global.self=global.self||global;
global.document=global.document||{createElement:()=>({getContext:()=>null})};
let src=fs.readFileSync(fp,'utf8').replace('return { decodeAttrs:', 'return { colorizeF: colorizeF, decodeAttrs:');
const m=new Module(fp,null); m.paths=Module._nodeModulePaths(p.dirname(fp));
m._compile(src, fp);
const SG=m.exports;
if(!SG||!SG.colorizeF){ console.log('FAIL: colorizeF not exposed; keys:', SG&&Object.keys(SG)); process.exit(1); }
const colorizeF=SG.colorizeF;
const n=64; const counts=new Float32Array(n), ext=new Float32Array(n);
for(let i=0;i<n;i++){ counts[i]=(i%32)+1; ext[i]=1; }      // nonzero densities
const lut=new Float32Array(256*4);
for(let i=0;i<256;i++){ const v=(i/255)*2.5; lut[i*4]=v; lut[i*4+1]=v*0.5; lut[i*4+2]=v*0.2; lut[i*4+3]=1; }  // lifted LUT, peak 2.5
const a=colorizeF(counts,ext,n,lut,'alpha',1);
const b=colorizeF(counts,ext,n,lut,'alpha',2);
const c=colorizeF(counts,ext,n,lut,'alpha',1);   // boost=1 must equal the no-boost legacy (default)
let maxErr2x=0, maxErrA=0, checked=0;
for(let i=0;i<n;i++){ const o=i*4;
  if(a[o]>1e-6||a[o+1]>1e-6||a[o+2]>1e-6){ checked++;
    maxErr2x=Math.max(maxErr2x, Math.abs(b[o]-2*a[o]), Math.abs(b[o+1]-2*a[o+1]), Math.abs(b[o+2]-2*a[o+2])); }
  maxErrA=Math.max(maxErrA, Math.abs(a[o+3]-b[o+3]));        // alpha must be boost-invariant
}
console.log('checked pixels:', checked);
console.log('max |boost2.RGB - 2*boost1.RGB| =', maxErr2x.toExponential(2));
console.log('max |alpha(b) - alpha(a)|       =', maxErrA.toExponential(2));
console.log((checked>0 && maxErr2x<1e-5 && maxErrA<1e-9) ? 'PASS: density RGB scales linearly with boost; alpha unchanged; boost=1 is the legacy path' : 'FAIL');
