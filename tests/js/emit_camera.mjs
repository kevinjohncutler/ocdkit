/* Emit an orbit-camera invViewProj from the SHIPPED mat4.js, so the Python
 * wgpu-native test can verify the JS camera composes with raymarch.wgsl under
 * perspective. Usage: node emit_camera.mjs [yaw] [pitch]  -> JSON on stdout. */
import { createRequire } from "node:module";
import path from "node:path";
const require = createRequire(import.meta.url);
const here = path.dirname(new URL(import.meta.url).pathname);
const M = require(path.join(here, "../../src/ocdkit/viewer/web/js/mat4.js"));

const yaw = parseFloat(process.argv[2] ?? "0");
const pitch = parseFloat(process.argv[3] ?? "0");
const NX = 12, NY = 12, NZ = 12, zScale = 1;
const sx = NX, sy = NY, sz = NZ * zScale;
const box = { min: [-sx / 2, -sy / 2, -sz / 2], max: [sx / 2, sy / 2, sz / 2] };
const radius = Math.max(sx, sy, sz) * 1.6;
const cam = M.orbitCamera({
  target: [0, 0, 0], up: [0, 1, 0], radius, yaw, pitch,
  fovy: Math.PI / 4, aspect: 1, near: 0.05, far: radius * 4,
});
process.stdout.write(JSON.stringify({ invViewProj: cam.invViewProj, eye: cam.eye, box, dims: [NX, NY, NZ] }));
