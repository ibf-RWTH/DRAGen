'use strict';

/* 3D view of a run: hand-written WebGL2, no dependencies.

   Geometry comes from /api/jobs/<id>/surface (see dragen/gui/surface.py): indexed triangles with
   (front grain, back region) per triangle. The front face shows the front grain, the back face the
   region behind it (nothing for the box), so a grain is always drawn in its own colour, seen from
   outside. Clip planes keep the part below x/y/z; the cut is closed by caps textured with the
   label slice at that plane. Colours and visibility per grain live in a palette texture
   (rgb = colour, a = 1 visible, 0.5 ghost, 0 hidden), so recolouring and isolating never touch
   the geometry. Picking renders grain ids into an offscreen framebuffer.

   Uses grainColor, hexToRgb, PHASE_COLORS, PHASE_NAMES and api() from app.js. */

const View3D = (() => {
  const PAL_W = 2048;                       // WebGL2 guarantees textures of 2048
  let gl = null, canvas = null, failed = null;
  let progSurf, progCap, locSurf, locCap;
  let vao = null, nVerts = 0, capVao = null, capBuf = null;
  let palTex = null, pickFbo = null, pickTex = null, pickDepth = null, pickSize = [0, 0];
  const capTex = [null, null, null], capIndex = [-1, -1, -1];
  let geo = null;              // {header, regions, nTri}
  const cam = { target: [0, 0, 0], dist: 10, yaw: -0.8, pitch: 0.5, fov: 35 * Math.PI / 180 };
  const opt = { color: 'grain', boxFaces: true, edges: false, clip: [null, null, null], isolate: null, ghost: true };
  let ctxRun = null;           // {jobId, detail, spacing, shape, size}
  let pending = false, loadToken = 0;

  /* ---------------------------------------------------------------------------- shaders */
  const SURF_VS = `#version 300 es
in vec3 aPos; in ivec2 aReg; in vec4 aBary;
uniform mat4 uMVP;
out vec3 vPos; flat out ivec2 vReg; out vec3 vBary;
void main() { vPos = aPos; vReg = aReg; vBary = aBary.xyz; gl_Position = uMVP * vec4(aPos, 1.0); }`;

  const COMMON_FS = `
uniform sampler2D uPal;
uniform vec3 uEye; uniform int uMode;
out vec4 outColor;
vec4 pal(int g) { return texelFetch(uPal, ivec2(g % ${PAL_W}, g / ${PAL_W}), 0); }
vec4 pickColor(int g) { return vec4(float(g & 255), float((g >> 8) & 255), float((g >> 16) & 255), 255.0) / 255.0; }
vec3 shade(vec3 base, vec3 n, vec3 p) {
  vec3 v = normalize(uEye - p);
  if (dot(n, v) < 0.0) n = -n;
  vec3 l = normalize(v + vec3(0.25, 0.15, 0.55));
  float diff = max(dot(n, l), 0.0);
  float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
  return base * (0.36 + 0.64 * diff) + 0.06 * rim;
}`;

  const SURF_FS = `#version 300 es
precision highp float; precision highp int;
in vec3 vPos; flat in ivec2 vReg; in vec3 vBary;
uniform vec3 uClip; uniform bool uBox; uniform bool uEdges;
${COMMON_FS}
void main() {
  if (any(greaterThanEqual(vPos, uClip))) discard;
  int g = gl_FrontFacing ? vReg.x : vReg.y;
  if (g <= 0) discard;
  if (!uBox && vReg.y < 0) discard;
  vec4 c = pal(g);
  if (uMode != 1 && c.a < 0.75) discard;
  if (uMode == 1 && (c.a < 0.1 || c.a >= 0.75)) discard;
  if (uMode == 2) { outColor = pickColor(g); return; }
  vec3 n = normalize(cross(dFdx(vPos), dFdy(vPos)));
  vec3 col = shade(c.rgb, n, vPos);
  if (uEdges && uMode == 0) {
    float e = min(min(vBary.x, vBary.y), vBary.z);
    float line = 1.0 - smoothstep(0.0, 1.4 * fwidth(e), e);
    col = mix(col, col * 0.22, line);
  }
  outColor = vec4(col, uMode == 1 ? 0.26 : 1.0);
}`;

  const CAP_VS = `#version 300 es
in vec3 aPos; uniform mat4 uMVP; out vec3 vPos;
void main() { vPos = aPos; gl_Position = uMVP * vec4(aPos, 1.0); }`;

  const CAP_FS = `#version 300 es
precision highp float; precision highp int; precision highp isampler2D;
in vec3 vPos;
uniform isampler2D uLab; uniform int uAxis; uniform ivec2 uRC; uniform vec3 uSpacing; uniform vec3 uClip;
uniform ivec2 uDims;
${COMMON_FS}
void main() {
  vec3 c3 = uClip; c3[uAxis] = 1e30;
  if (any(greaterThanEqual(vPos, c3))) discard;
  int row = int(floor(vPos[uRC.x] / uSpacing[uRC.x]));
  int col = int(floor(vPos[uRC.y] / uSpacing[uRC.y]));
  row = clamp(row, 0, uDims.y - 1); col = clamp(col, 0, uDims.x - 1);
  int g = texelFetch(uLab, ivec2(col, row), 0).r;
  if (g <= 0) discard;
  vec4 c = pal(g);
  if (c.a < 0.75) discard;
  if (uMode == 2) { outColor = pickColor(g); return; }
  vec3 n = vec3(0.0); n[uAxis] = 1.0;
  outColor = vec4(shade(c.rgb * 0.92, n, vPos), 1.0);
}`;

  function compile(vs, fs) {
    const make = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, make(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, make(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  function uniforms(prog, names) {
    const out = {};
    for (const n of names) out[n] = gl.getUniformLocation(prog, n);
    return out;
  }

  /* ---------------------------------------------------------------------------- matrices */
  const m4 = {
    mul(a, b) {
      const o = new Float32Array(16);
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
        o[c * 4 + r] = s;
      }
      return o;
    },
    persp(fov, aspect, near, far) {
      const f = 1 / Math.tan(fov / 2), nf = 1 / (near - far);
      return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
    },
    lookAt(eye, at, up) {
      const z = norm(sub(eye, at)), x = norm(cross(up, z)), y = cross(z, x);
      return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
    },
  };
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = a => { const l = Math.hypot(...a) || 1; return scale(a, 1 / l); };

  function eye() {
    const cp = Math.cos(cam.pitch);
    return add(cam.target, scale([cp * Math.cos(cam.yaw), cp * Math.sin(cam.yaw), Math.sin(cam.pitch)], cam.dist));
  }

  function mvp() {
    const aspect = canvas.width / Math.max(1, canvas.height);
    const view = m4.lookAt(eye(), cam.target, [0, 0, 1]);
    const proj = m4.persp(cam.fov, aspect, cam.dist / 200, cam.dist * 20);
    return m4.mul(proj, view);
  }

  /* ---------------------------------------------------------------------------- set-up */
  function init(el) {
    canvas = el;
    gl = canvas.getContext('webgl2', { antialias: true, preserveDrawingBuffer: false });
    if (!gl) { failed = 'WebGL2 is not available in this browser.'; return false; }
    try {
      progSurf = compile(SURF_VS, SURF_FS);
      progCap = compile(CAP_VS, CAP_FS);
    } catch (err) { failed = 'WebGL shaders failed: ' + err.message; gl = null; return false; }
    locSurf = uniforms(progSurf, ['uMVP', 'uClip', 'uBox', 'uEdges', 'uPal', 'uEye', 'uMode']);
    locCap = uniforms(progCap, ['uMVP', 'uClip', 'uLab', 'uAxis', 'uRC', 'uSpacing', 'uDims', 'uPal', 'uEye', 'uMode']);
    palTex = gl.createTexture();
    capVao = gl.createVertexArray();
    capBuf = gl.createBuffer();
    gl.bindVertexArray(capVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, capBuf);
    gl.enableVertexAttribArray(gl.getAttribLocation(progCap, 'aPos'));
    gl.vertexAttribPointer(gl.getAttribLocation(progCap, 'aPos'), 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    new ResizeObserver(() => requestRender()).observe(canvas.parentElement);
    bindInput();
    return true;
  }

  function parse(buffer) {
    const dv = new DataView(buffer);
    const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
    if (magic !== 'DRGS') throw new Error('unexpected surface payload');
    const hlen = dv.getUint32(4, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, hlen)));
    let off = 8 + hlen;
    const V = header.n_points, T = header.n_triangles;
    const points = new Float32Array(buffer, off, V * 3); off += V * 12;
    const tris = new Uint32Array(buffer, off, T * 3); off += T * 12;
    const regions = new Int32Array(buffer, off, T * 2); off += T * 8;
    const mask = new Uint8Array(buffer, off, T);
    return { header, points, tris, regions, mask };
  }

  function upload(g) {
    const T = g.header.n_triangles;
    const pos = new Float32Array(T * 9), reg = new Int32Array(T * 6), bary = new Uint8Array(T * 12);
    const { points, tris, regions, mask } = g;
    for (let t = 0; t < T; t++) {
      const m = mask[t];
      for (let k = 0; k < 3; k++) {
        const v = tris[3 * t + k], o = 9 * t + 3 * k;
        pos[o] = points[3 * v]; pos[o + 1] = points[3 * v + 1]; pos[o + 2] = points[3 * v + 2];
        reg[6 * t + 2 * k] = regions[2 * t]; reg[6 * t + 2 * k + 1] = regions[2 * t + 1];
        const b = 12 * t + 4 * k;
        for (let j = 0; j < 3; j++) bary[b + j] = (j === k || (m >> j) & 1) ? 255 : 0;
      }
    }
    if (vao) gl.deleteVertexArray(vao);
    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const attr = (name, data, size, type, integer, normalized = false) => {
      const loc = gl.getAttribLocation(progSurf, name);
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      if (integer) gl.vertexAttribIPointer(loc, size, type, 0, 0);
      else gl.vertexAttribPointer(loc, size, type, normalized, 0, 0);
    };
    attr('aPos', pos, 3, gl.FLOAT, false);
    attr('aReg', reg, 2, gl.INT, true);
    attr('aBary', bary, 4, gl.UNSIGNED_BYTE, false, true);
    gl.bindVertexArray(null);
    nVerts = T * 3;
  }

  /* ---------------------------------------------------------------------------- palette */
  function palette() {
    if (!geo || !ctxRun) return;
    const meta = ctxRun.detail.grains || { phase: [] };
    const n = Math.max(meta.phase.length, maxGrain() + 1, 2);
    const rows = Math.ceil(n / PAL_W);
    const data = new Uint8Array(PAL_W * rows * 4);
    const iso = opt.isolate;
    const neigh = iso ? neighbours(iso) : null;
    for (let id = 1; id < n; id++) {
      let c;
      if (opt.color === 'phase') {
        const base = hexToRgb(PHASE_COLORS[meta.phase[id]] || '#888888');
        const s = 0.82 + 0.18 * ((id * 0.618) % 1);
        c = base.map(v => Math.round(v * s));
      } else c = grainColor(id);
      let a = 255;
      if (iso) a = id === iso ? 255 : (neigh.has(id) && opt.ghost ? 128 : 0);
      data.set([c[0], c[1], c[2], a], id * 4);
    }
    gl.bindTexture(gl.TEXTURE_2D, palTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, PAL_W, rows, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    requestRender();
  }

  function maxGrain() {
    let m = 0;
    const r = geo.regions;
    for (let i = 0; i < r.length; i++) if (r[i] > m) m = r[i];
    return m;
  }

  function neighbours(g) {
    const out = new Set();
    const r = geo.regions;
    for (let t = 0; t < r.length; t += 2) {
      if (r[t] === g && r[t + 1] > 0) out.add(r[t + 1]);
      else if (r[t + 1] === g) out.add(r[t]);
    }
    return out;
  }

  /* ---------------------------------------------------------------------------- caps */
  async function updateCap(axis) {
    const k = opt.clip[axis];
    if (k === null || k >= ctxRun.shape[axis] || k <= 0) { capIndex[axis] = -1; requestRender(); return; }
    if (capIndex[axis] === k - 1) return;
    const token = loadToken;
    const res = await api(`jobs/${encodeURIComponent(ctxRun.jobId)}/slice?axis=${axis}&index=${k - 1}`);
    if (token !== loadToken || opt.clip[axis] !== k) return;
    const raw = atob(res.data);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    if (!capTex[axis]) capTex[axis] = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, capTex[axis]);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32I, res.cols, res.rows, 0, gl.RED_INTEGER, gl.INT, new Int32Array(bytes.buffer));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    capTex[axis].dims = [res.cols, res.rows];
    capIndex[axis] = k - 1;
    requestRender();
  }

  function clipVector() {
    return [0, 1, 2].map(a => (opt.clip[a] === null || opt.clip[a] >= ctxRun.shape[a]) ? 1e30
      : opt.clip[a] * ctxRun.spacing[a] - 1e-4 * ctxRun.spacing[a]);
  }

  /* ---------------------------------------------------------------------------- drawing */
  function requestRender() {
    if (pending || !gl) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; draw(); });
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

  function draw(mode = 'screen') {
    if (!gl || !geo) return;
    resize();
    const pick = mode === 'pick';
    if (pick) ensurePickTarget();
    gl.bindFramebuffer(gl.FRAMEBUFFER, pick ? pickFbo : null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    const bg = pick ? [0, 0, 0] : hexToRgb(getComputedStyle(document.documentElement).getPropertyValue('--surface-2').trim() || '#202020');
    gl.clearColor(bg[0] / 255, bg[1] / 255, bg[2] / 255, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
    gl.depthMask(true);
    const M = mvp(), E = eye(), clip = clipVector();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, palTex);

    // caps
    gl.useProgram(progCap);
    gl.uniformMatrix4fv(locCap.uMVP, false, M);
    gl.uniform3fv(locCap.uClip, clip);
    gl.uniform3fv(locCap.uEye, E);
    gl.uniform3fv(locCap.uSpacing, ctxRun.spacing);
    gl.uniform1i(locCap.uPal, 0);
    gl.uniform1i(locCap.uLab, 1);
    gl.uniform1i(locCap.uMode, pick ? 2 : 0);
    const size = ctxRun.size;
    for (let a = 0; a < 3; a++) {
      if (capIndex[a] < 0 || !capTex[a]) continue;
      const b = (a + 1) % 3, c = (a + 2) % 3;
      const p = opt.clip[a] * ctxRun.spacing[a];
      const q = (u, v) => { const o = [0, 0, 0]; o[a] = p; o[b] = u; o[c] = v; return o; };
      const quad = [q(0, 0), q(size[b], 0), q(size[b], size[c]), q(0, 0), q(size[b], size[c]), q(0, size[c])].flat();
      gl.bindVertexArray(capVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, capBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(quad), gl.DYNAMIC_DRAW);
      const rc = [[1, 2], [0, 2], [0, 1]][a];
      gl.uniform1i(locCap.uAxis, a);
      gl.uniform2i(locCap.uRC, rc[0], rc[1]);
      gl.uniform2i(locCap.uDims, capTex[a].dims[0], capTex[a].dims[1]);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, capTex[a]);
      gl.activeTexture(gl.TEXTURE0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    // surfaces: opaque, then ghosts
    gl.useProgram(progSurf);
    gl.bindVertexArray(vao);
    gl.uniformMatrix4fv(locSurf.uMVP, false, M);
    gl.uniform3fv(locSurf.uClip, clip);
    gl.uniform3fv(locSurf.uEye, E);
    gl.uniform1i(locSurf.uBox, opt.boxFaces ? 1 : 0);
    gl.uniform1i(locSurf.uEdges, opt.edges ? 1 : 0);
    gl.uniform1i(locSurf.uPal, 0);
    gl.uniform1i(locSurf.uMode, pick ? 2 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, nVerts);
    if (!pick && opt.isolate && opt.ghost) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      gl.uniform1i(locSurf.uMode, 1);
      gl.drawArrays(gl.TRIANGLES, 0, nVerts);
      gl.depthMask(true);
      gl.disable(gl.BLEND);
    }
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  function ensurePickTarget() {
    if (pickFbo && pickSize[0] === canvas.width && pickSize[1] === canvas.height) return;
    if (!pickFbo) { pickFbo = gl.createFramebuffer(); pickTex = gl.createTexture(); pickDepth = gl.createRenderbuffer(); }
    gl.bindTexture(gl.TEXTURE_2D, pickTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, canvas.width, canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindRenderbuffer(gl.RENDERBUFFER, pickDepth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, canvas.width, canvas.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, pickFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pickTex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, pickDepth);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    pickSize = [canvas.width, canvas.height];
  }

  function pickAt(clientX, clientY) {
    if (!gl || !geo) return 0;
    draw('pick');
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((clientX - rect.left) / rect.width * canvas.width);
    const y = canvas.height - 1 - Math.floor((clientY - rect.top) / rect.height * canvas.height);
    const px = new Uint8Array(4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, pickFbo);
    gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return px[0] + (px[1] << 8) + (px[2] << 16);
  }

  /* ---------------------------------------------------------------------------- input */
  function bindInput() {
    let drag = null, lastHover = 0;
    const pointers = new Map();
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('pointerdown', e => {
      canvas.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      drag = { x: e.clientX, y: e.clientY, moved: false, pan: e.button === 2 || e.button === 1 || e.shiftKey };
    });
    canvas.addEventListener('pointermove', e => {
      if (!pointers.has(e.pointerId)) {
        const now = performance.now();
        if (now - lastHover > 60) { lastHover = now; hover(e); }
        return;
      }
      const prev = pointers.get(e.pointerId);
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const other = [...pointers.entries()].find(([id]) => id !== e.pointerId)[1];
        const before = Math.hypot(prev[0] - other[0], prev[1] - other[1]);
        const after = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (before > 0) cam.dist *= before / after;
        pan((e.clientX - prev[0]) / 2, (e.clientY - prev[1]) / 2);
      } else if (drag) {
        const dx = e.clientX - prev[0], dy = e.clientY - prev[1];
        if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 3) drag.moved = true;
        if (drag.pan) pan(dx, dy);
        else {
          cam.yaw -= dx * 0.008;
          cam.pitch = Math.max(-1.55, Math.min(1.55, cam.pitch + dy * 0.008));
        }
      }
      hideTip();
      requestRender();
    });
    const end = e => {
      if (drag && !drag.moved && pointers.size === 1) click(e);
      pointers.delete(e.pointerId);
      if (!pointers.size) drag = null;
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', e => { pointers.delete(e.pointerId); drag = null; });
    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      cam.dist *= Math.exp(e.deltaY * 0.0012);
      requestRender();
    }, { passive: false });
    canvas.addEventListener('dblclick', e => {
      const g = pickAt(e.clientX, e.clientY);
      if (g) isolate(g);
    });
    canvas.addEventListener('pointerleave', hideTip);
  }

  function pan(dx, dy) {
    const e = eye(), f = norm(sub(cam.target, e)), right = norm(cross(f, [0, 0, 1])), up = cross(right, f);
    const s = cam.dist * Math.tan(cam.fov / 2) * 2 / Math.max(1, canvas.clientHeight);
    cam.target = add(cam.target, add(scale(right, -dx * s), scale(up, dy * s)));
  }

  function grainInfo(g) {
    const meta = ctxRun.detail.grains || { phase: [], volume: [], actual_volume: [] };
    const phase = PHASE_NAMES[meta.phase[g]] || '';
    const vol = meta.actual_volume && meta.actual_volume[g] !== undefined ? meta.actual_volume[g].toFixed(1) + ' µm³' : '';
    const target = meta.volume && meta.volume[g] ? ' (target ' + meta.volume[g].toFixed(1) + ')' : '';
    return { phase, text: `grain ${g}${phase ? ' · ' + phase : ''}${vol ? ' · ' + vol + target : ''}` };
  }

  function hover(e) {
    if (!geo) return;
    const g = pickAt(e.clientX, e.clientY);
    const tip = document.getElementById('gl-tip');
    if (!g) return hideTip();
    tip.textContent = grainInfo(g).text;
    const wrap = canvas.parentElement.getBoundingClientRect();
    tip.style.left = (e.clientX - wrap.left) + 'px';
    tip.style.top = (e.clientY - wrap.top) + 'px';
    tip.classList.remove('hidden');
  }

  function hideTip() { document.getElementById('gl-tip')?.classList.add('hidden'); }

  function click(e) {
    const g = pickAt(e.clientX, e.clientY);
    showSelection(g);
  }

  function showSelection(g) {
    const panel = document.getElementById('gl-pick');
    if (!g) { panel.classList.add('hidden'); return; }
    const info = grainInfo(g);
    const n = neighbours(g).size;
    panel.replaceChildren(
      el('div', { class: 'pick-title' }, el('i', { style: `background:rgb(${grainColor(g).join(',')})` }), `Grain ${g}`),
      el('div', { class: 'muted' }, info.text.replace(/^grain \d+ · /, '')),
      el('div', { class: 'muted' }, `${n} neighbour${n === 1 ? '' : 's'}`),
      el('div', { class: 'pick-actions' },
        el('button', { class: 'btn small', onclick: () => isolate(g) }, 'Isolate'),
        opt.isolate ? el('button', { class: 'btn small ghost', onclick: () => isolate(null) }, 'Show all') : null));
    panel.classList.remove('hidden');
  }

  function isolate(g) {
    opt.isolate = g;
    palette();
    showSelection(g);
    document.getElementById('gl-showall').disabled = !g;
  }

  /* ---------------------------------------------------------------------------- run glue */
  function resetView() {
    if (!ctxRun) return;
    const s = ctxRun.size;
    cam.target = scale(s, 0.5);
    cam.dist = Math.hypot(...s) * 1.35;
    cam.yaw = -0.8; cam.pitch = 0.5;
    requestRender();
  }

  async function load(jobId, detail, source, step) {
    const token = ++loadToken;
    const status = document.getElementById('gl-status');
    const busy = document.getElementById('gl-busy');
    busy.classList.remove('hidden');
    try {
      const t0 = performance.now();
      const mesh = source.startsWith('smoothed:') ? source.slice(9) : null;
      const q = mesh ? `source=smoothed&mesh=${encodeURIComponent(mesh)}` : `source=staircase&step=${step}`;
      const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/surface?${q}`);
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
      const buffer = await res.arrayBuffer();
      if (token !== loadToken) return;
      const g = parse(buffer);
      const result = detail.result;
      const spacing = detail.spacing || result.spacing;
      const shape = result.shape;
      ctxRun = { jobId, detail, spacing, shape, size: shape.map((n, i) => n * spacing[i]) };
      geo = { header: g.header, regions: g.regions };
      upload(g);
      opt.isolate = null;
      capIndex.fill(-1);
      palette();
      for (let a = 0; a < 3; a++) updateCap(a);
      if (!cam.loaded) { resetView(); cam.loaded = true; }
      const h = g.header;
      const mb = (buffer.byteLength / 1e6).toFixed(1);
      const parts = [`${h.n_triangles.toLocaleString()} triangles`, `${mb} MB`, `${Math.round(performance.now() - t0)} ms`];
      if (h.source === 'staircase') {
        parts.splice(1, 0, `merged from ${h.unmerged_triangles.toLocaleString()}`);
        if (h.step > 1) parts.push(`downsampled ×${h.step}`);
      }
      status.textContent = parts.join(' · ');
      document.getElementById('gl-pick').classList.add('hidden');
      requestRender();
    } catch (err) {
      status.textContent = '';
      toast(err.message, true);
    } finally {
      if (token === loadToken) busy.classList.add('hidden');
    }
  }

  let uiReady = false;
  function initUi() {
    if (uiReady) return;
    uiReady = true;
    const c = document.getElementById('gl-canvas');
    if (!init(c)) {
      document.getElementById('gl-wrap').replaceChildren(el('div', { class: 'gl-fallback' },
        el('b', {}, '3D view unavailable'), el('div', {}, failed), el('div', { class: 'muted' }, 'The slice viewer works without WebGL.')));
      return;
    }
    const reload = () => current && load(current.jobId, current.detail, document.getElementById('gl-source').value,
      document.getElementById('gl-step').value);
    document.getElementById('gl-source').addEventListener('change', () => {
      document.getElementById('gl-step').disabled = document.getElementById('gl-source').value !== 'staircase';
      reload();
    });
    document.getElementById('gl-step').addEventListener('change', reload);
    document.querySelectorAll('#gl-color button').forEach(b => b.addEventListener('click', () => {
      document.querySelectorAll('#gl-color button').forEach(x => x.classList.toggle('active', x === b));
      opt.color = b.dataset.color;
      palette();
    }));
    document.getElementById('gl-box').addEventListener('change', e => { opt.boxFaces = e.target.checked; requestRender(); });
    document.getElementById('gl-edges').addEventListener('change', e => { opt.edges = e.target.checked; requestRender(); });
    document.getElementById('gl-ghost').addEventListener('change', e => { opt.ghost = e.target.checked; palette(); });
    document.getElementById('gl-reset').addEventListener('click', resetView);
    document.getElementById('gl-showall').addEventListener('click', () => isolate(null));
    for (const a of [0, 1, 2]) {
      const slider = document.getElementById('gl-clip-' + a);
      let timer = null;
      slider.addEventListener('input', () => {
        if (!ctxRun) return;
        const k = Number(slider.value);
        opt.clip[a] = k >= Number(slider.max) ? null : k;
        document.getElementById('gl-clip-label-' + a).textContent = opt.clip[a] === null ? 'off'
          : (k * ctxRun.spacing[a]).toFixed(1) + ' µm';
        requestRender();
        clearTimeout(timer);
        timer = setTimeout(() => updateCap(a), 50);
      });
    }
  }

  let current = null;
  function open(jobId, detail) {
    initUi();
    if (!gl) return;
    const same = current && current.jobId === jobId;
    current = { jobId, detail };
    const sources = (detail.surfaces || []).map(s => s.source === 'smoothed'
      ? { value: 'smoothed:' + s.mesh, label: s.label } : { value: 'staircase', label: s.label });
    const select = document.getElementById('gl-source');
    const before = select.value;
    select.replaceChildren(...sources.map(s => el('option', { value: s.value }, s.label)));
    if (sources.some(s => s.value === before)) select.value = before;
    document.getElementById('gl-step').disabled = select.value !== 'staircase';
    if (same && geo) { requestRender(); return; }
    cam.loaded = false;
    opt.clip = [null, null, null];
    const shape = detail.result.shape;
    for (const a of [0, 1, 2]) {
      const s = document.getElementById('gl-clip-' + a);
      s.max = shape[a];
      s.value = shape[a];
      document.getElementById('gl-clip-label-' + a).textContent = 'off';
    }
    document.getElementById('gl-showall').disabled = true;
    load(jobId, detail, select.value, document.getElementById('gl-step').value);
  }

  function forget() {
    current = null;
    loadToken++;
    geo = null;
    ctxRun = null;
  }

  return { open, forget, requestRender, available: () => !!gl, isolate };
})();
