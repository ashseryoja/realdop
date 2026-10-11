import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

// На телефонах — облегчённая модель (вдвое меньше полигонов и веса)
const LITE_QUERY = '(max-width: 768px)';
const MODEL_URL = window.matchMedia(LITE_QUERY).matches
  ? 'assets/models/realdop-lite.glb'
  : 'assets/models/realdop.glb';
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const canvas = document.querySelector('.hero__canvas');
const soon = document.querySelector('.hero__soon');

/* ------------------------------------------------------------------ */
/* GLSL                                                                */
/* ------------------------------------------------------------------ */

// 2D simplex noise — Ashima Arts / Stefan Gustavson (MIT)
const SIMPLEX_2D = /* glsl */ `
vec3 permute3(vec3 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
float snoise2(vec2 v) {
  const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
  vec2 i = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute3(permute3(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
  vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
  m = m * m; m = m * m;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
  vec3 g;
  g.x = a0.x * x0.x + h.x * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}
`;

// 3D simplex noise — Ashima Arts / Stefan Gustavson (MIT)
const SIMPLEX_3D = /* glsl */ `
vec4 permute4(vec4 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
vec4 taylorInvSqrt4(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
float snoise3(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod(i, 289.0);
  vec4 p = permute4(permute4(permute4(
    i.z + vec4(0.0, i1.z, i2.z, 1.0)) +
    i.y + vec4(0.0, i1.y, i2.y, 1.0)) +
    i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 1.0 / 7.0;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt4(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}
`;

const FULLSCREEN_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

// Небо с облаками. Рендерится в уменьшенный буфер — облака мягкие, детализация не страдает.
const SKY_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform float uTime;
uniform vec2 uRes;
uniform vec2 uParallax;

${SIMPLEX_2D}

float fbm(vec2 p) {
  float f = 0.0;
  float a = 0.5;
  mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
  for (int i = 0; i < 6; i++) {
    f += a * snoise2(p);
    p = m * p;
    a *= 0.5;
  }
  return f;
}

float cloudField(vec2 p, float t) {
  vec2 q = vec2(
    fbm(p * 0.7 + vec2(0.0, t * 0.03)),
    fbm(p * 0.7 + vec2(5.2, 1.3) - vec2(t * 0.025, 0.0))
  );
  return fbm(p + 0.55 * q) * 0.5 + 0.5;
}

void main() {
  float aspect = uRes.x / uRes.y;
  vec2 uv = vUv;
  vec2 p = (uv - 0.5) * vec2(aspect, 1.0);
  float t = uTime;

  // Небо: насыщенный голубой, темнее к верху
  vec3 top = vec3(0.110, 0.408, 0.776);
  vec3 mid = vec3(0.204, 0.525, 0.851);
  vec3 bot = vec3(0.318, 0.612, 0.898);
  vec3 sky = mix(bot, mid, smoothstep(0.0, 0.55, uv.y));
  sky = mix(sky, top, smoothstep(0.55, 1.0, uv.y));

  // Облака обрамляют экран, центр под логотипом остаётся чистым
  vec2 e = abs(uv - 0.5) * 2.0;
  float edge = pow(pow(e.x, 3.0) + pow(e.y, 3.0), 1.0 / 3.0);
  float bias = (edge - 0.82) * 0.95;

  float zoom = 1.25 * max(1.0, 0.75 / aspect);
  vec2 cp = p * zoom + uParallax * 0.04 + vec2(t * 0.012, -t * 0.004);
  float d = cloudField(cp, t) + bias;
  float detail = snoise2(cp * 9.0 + t * 0.05) * 0.5 + snoise2(cp * 19.0 - t * 0.04) * 0.25;
  float shape = d + detail * 0.09;
  float cover = smoothstep(0.46, 0.74, shape);

  // Объём: сравниваем плотность со смещённой к свету точкой
  float dl = cloudField(cp + vec2(-0.03, 0.04), t) + bias;
  float light = clamp((d - dl) * 6.0 + 0.7, 0.0, 1.0);
  vec3 cloud = mix(vec3(0.74, 0.82, 0.93), vec3(1.0), light);
  cloud = mix(cloud, vec3(1.0), smoothstep(0.7, 1.0, shape) * 0.7);

  // Голубая дымка по краям облаков
  vec3 col = mix(sky, mix(sky, vec3(0.9, 0.95, 1.0), 0.55), smoothstep(0.36, 0.6, shape) * 0.45);
  col = mix(col, cloud, cover);
  gl_FragColor = vec4(col, 1.0);
}
`;

const BACKDROP_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tSky;
void main() {
  gl_FragColor = vec4(texture2D(tSky, vUv).rgb, 1.0);
}
`;

// Плёночное зерно поверх всего кадра (мультипликативный бленд)
const GRAIN_FRAG = /* glsl */ `
precision highp float;
uniform float uTime;
uniform float uAmount;
float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
void main() {
  float n = hash(gl_FragCoord.xy + fract(uTime * 7.13) * 431.0);
  gl_FragColor = vec4(vec3(0.5 + (n - 0.5) * uAmount), 1.0);
}
`;

// Окружение для хромовых отражений: небо, облака, солнце и тёмная «земля» для контраста
const ENV_VERT = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = normalize(position);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const ENV_FRAG = /* glsl */ `
precision highp float;
varying vec3 vDir;

${SIMPLEX_3D}

float fbm3(vec3 p) {
  float f = 0.0;
  float a = 0.5;
  for (int i = 0; i < 5; i++) {
    f += a * snoise3(p);
    p *= 2.03;
    a *= 0.5;
  }
  return f;
}

void main() {
  vec3 d = normalize(vDir);
  float y = d.y;

  vec3 zenith = vec3(0.025, 0.14, 0.52);
  vec3 horizon = vec3(0.55, 0.68, 0.88);
  vec3 ground = vec3(0.012, 0.018, 0.03);

  vec3 col = y > 0.0
    ? mix(horizon, zenith, pow(y, 0.35))
    : mix(horizon * 0.7, ground, pow(-y, 0.22));

  // Облака: яркие пятна с чёткими краями — дают «хромовые» блики
  float c = fbm3(d * 2.4 + vec3(0.0, 0.0, 1.7)) * 0.5 + 0.5;
  c = smoothstep(0.5, 0.62, c) * smoothstep(-0.02, 0.15, y);
  col = mix(col, vec3(2.2), c);

  // Тёмные полосы для контраста в изгибах
  float bands = fbm3(d * 1.6 + vec3(4.0)) * 0.5 + 0.5;
  col *= mix(1.0, 0.25, smoothstep(0.55, 0.7, bands) * (1.0 - c));

  vec3 sunDir = normalize(vec3(0.45, 0.62, 0.65));
  float s = max(dot(d, sunDir), 0.0);
  col += vec3(1.0, 0.97, 0.92) * (pow(s, 900.0) * 80.0 + pow(s, 30.0) * 1.2);

  vec3 fillDir = normalize(vec3(-0.6, 0.3, 0.75));
  float f = max(dot(d, fillDir), 0.0);
  col += vec3(0.85, 0.92, 1.0) * pow(f, 80.0) * 10.0;

  gl_FragColor = vec4(col, 1.0);
}
`;

// Жидкая деформация: шум по нормали + мягкая волна + реакция на курсор и тап
const LIQUID_VERT_HEAD = /* glsl */ `
uniform float uTime;
uniform float uAmp;
uniform vec2 uPointer;
uniform float uPointerAmp;
uniform vec3 uRipple;
uniform float uRippleAmp;

${SIMPLEX_3D}

vec3 liquidOffset(vec3 p, vec3 n) {
  float t = uTime;
  float big = snoise3(vec3(p.xy * 1.15, t * 0.22));
  float small = snoise3(p * 3.4 + vec3(0.0, t * 0.42, t * 0.3));
  float d = (big * 0.6 + small * 0.4) * uAmp;

  float pd = length(p.xy - uPointer);
  d += exp(-pd * pd * 16.0) * uPointerAmp;

  float age = t - uRipple.z;
  float rd = length(p.xy - uRipple.xy);
  float front = rd - age * 0.95;
  d += sin(rd * 16.0 - age * 9.0) * exp(-front * front * 14.0) * exp(-age * 1.3) * uRippleAmp;

  vec3 o = n * d;
  o.z += sin(p.x * 2.1 + t * 0.85) * 0.022 + sin(p.y * 2.6 - t * 0.7) * 0.016;
  o.y += sin(p.x * 2.9 + t * 1.05) * 0.010;
  o.x += sin(p.y * 2.3 + t * 0.9) * 0.006;
  return o;
}
`;

const LIQUID_VERT_BODY = /* glsl */ `
#include <beginnormal_vertex>
vec3 lqT = normalize(cross(objectNormal, abs(objectNormal.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
vec3 lqB = normalize(cross(objectNormal, lqT));
vec3 lqP0 = position + liquidOffset(position, objectNormal);
vec3 lqP1 = position + lqT * 0.01;
lqP1 += liquidOffset(lqP1, objectNormal);
vec3 lqP2 = position + lqB * 0.01;
lqP2 += liquidOffset(lqP2, objectNormal);
vec3 lqN = normalize(cross(lqP1 - lqP0, lqP2 - lqP0));
objectNormal = dot(lqN, objectNormal) < 0.0 ? -lqN : lqN;
`;

/* ------------------------------------------------------------------ */
/* Setup                                                               */
/* ------------------------------------------------------------------ */

function createRenderer() {
  try {
    const r = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    return r;
  } catch (err) {
    console.warn('WebGL недоступен:', err);
    return null;
  }
}

const renderer = createRenderer();
if (!renderer) {
  document.documentElement.classList.add('no-webgl');
} else {
  init(renderer);
}

function init(renderer) {
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const clock = new THREE.Clock();
  const shared = {
    uTime: { value: 0 },
  };

  /* ---- Небо ---- */
  const quad = new THREE.PlaneGeometry(2, 2);
  const orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  const skyTarget = new THREE.WebGLRenderTarget(2, 2, { depthBuffer: false });
  const skyMat = new THREE.ShaderMaterial({
    vertexShader: FULLSCREEN_VERT,
    fragmentShader: SKY_FRAG,
    uniforms: {
      uTime: shared.uTime,
      uRes: { value: new THREE.Vector2(1, 1) },
      uParallax: { value: new THREE.Vector2() },
    },
    depthTest: false,
    depthWrite: false,
  });
  const skyScene = new THREE.Scene();
  skyScene.add(new THREE.Mesh(quad, skyMat));

  const scene = new THREE.Scene();

  const backdrop = new THREE.Mesh(
    quad,
    new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERT,
      fragmentShader: BACKDROP_FRAG,
      uniforms: { tSky: { value: skyTarget.texture } },
      depthTest: false,
      depthWrite: false,
    }),
  );
  backdrop.frustumCulled = false;
  backdrop.renderOrder = -10;
  scene.add(backdrop);

  const grain = new THREE.Mesh(
    quad,
    new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERT,
      fragmentShader: GRAIN_FRAG,
      uniforms: { uTime: shared.uTime, uAmount: { value: 0.07 } },
      depthTest: false,
      depthWrite: false,
      transparent: true,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.DstColorFactor,
      blendDst: THREE.SrcColorFactor,
    }),
  );
  grain.frustumCulled = false;
  grain.renderOrder = 100;
  scene.add(grain);

  /* ---- Отражения ---- */
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.add(
    new THREE.Mesh(
      new THREE.SphereGeometry(10, 64, 32),
      new THREE.ShaderMaterial({
        vertexShader: ENV_VERT,
        fragmentShader: ENV_FRAG,
        side: THREE.BackSide,
        depthWrite: false,
      }),
    ),
  );
  scene.environment = pmrem.fromScene(envScene, 0, 0.1, 50).texture;
  pmrem.dispose();

  /* ---- Камера и логотип ---- */
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50);
  camera.position.set(0, 0, 6);

  const logo = new THREE.Group();
  scene.add(logo);

  const liquid = {
    uAmp: { value: 0.014 },
    uPointer: { value: new THREE.Vector2(10, 10) },
    uPointerAmp: { value: 0 },
    uRipple: { value: new THREE.Vector3(10, 10, -100) },
    uRippleAmp: { value: 0 },
  };

  const material = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    metalness: 1,
    roughness: 0.06,
    envMapIntensity: 1.0,
  });
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, liquid, { uTime: shared.uTime });
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${LIQUID_VERT_HEAD}`)
      .replace('#include <beginnormal_vertex>', LIQUID_VERT_BODY)
      .replace('#include <begin_vertex>', 'vec3 transformed = lqP0;');
  };

  const logoSize = new THREE.Vector3(2, 1.8, 0.16);
  let logoMesh = null;
  let glintSpots = [];
  const glints = [];

  /* ---- Раскладка ---- */
  const view = { width: 1, height: 1, scale: 1, baseY: 0, visW: 1, visH: 1 };

  function layout() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    view.width = w;
    view.height = h;

    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();

    const skyScale = Math.min(dpr, 1.5) * 0.6;
    skyTarget.setSize(Math.max(2, Math.round(w * skyScale)), Math.max(2, Math.round(h * skyScale)));
    skyMat.uniforms.uRes.value.set(w, h);

    const visH = 2 * camera.position.z * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const visW = visH * camera.aspect;
    view.visW = visW;
    view.visH = visH;

    const portrait = w / h < 0.8;
    const maxW = (portrait ? 0.86 : 0.7) * visW;
    const maxH = (portrait ? 0.48 : 0.62) * visH;
    view.scale = Math.min(maxW / logoSize.x, maxH / logoSize.y);
    view.baseY = (portrait ? 0.06 : 0.05) * visH;

    // Позиция надписи — сразу под проекцией логотипа
    camera.updateMatrixWorld();
    const bottomWorld = view.baseY - (logoSize.y / 2) * view.scale;
    const ndc = new THREE.Vector3(0, bottomWorld, 0).project(camera);
    soon.style.setProperty('--logo-bottom', `${((1 - ndc.y) / 2) * h}px`);
  }

  window.addEventListener('resize', layout);
  layout();

  /* ---- Указатель ---- */
  const pointer = { x: 0, y: 0, sx: 0, sy: 0, active: false, energy: 0, lastX: 0, lastY: 0 };
  const raycastPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  const ndcVec = new THREE.Vector2();
  const raycaster = new THREE.Raycaster();
  const hit = new THREE.Vector3();

  function toLogoLocal(clientX, clientY, out) {
    ndcVec.set((clientX / view.width) * 2 - 1, -(clientY / view.height) * 2 + 1);
    raycaster.setFromCamera(ndcVec, camera);
    if (!raycaster.ray.intersectPlane(raycastPlane, hit) || !logoMesh) return false;
    logoMesh.worldToLocal(hit);
    out.set(hit.x, hit.y);
    return true;
  }

  const localPointer = new THREE.Vector2();
  window.addEventListener('pointermove', (e) => {
    pointer.x = (e.clientX / view.width) * 2 - 1;
    pointer.y = -(e.clientY / view.height) * 2 + 1;
    const dx = e.clientX - pointer.lastX;
    const dy = e.clientY - pointer.lastY;
    pointer.lastX = e.clientX;
    pointer.lastY = e.clientY;
    pointer.energy = Math.min(1, pointer.energy + Math.hypot(dx, dy) / 120);
    pointer.active = true;
    if (toLogoLocal(e.clientX, e.clientY, localPointer)) liquid.uPointer.value.copy(localPointer);
  }, { passive: true });

  document.documentElement.addEventListener('pointerleave', () => { pointer.active = false; });
  window.addEventListener('pointerup', (e) => {
    if (e.pointerType !== 'mouse') pointer.active = false;
  }, { passive: true });

  const ripplePoint = new THREE.Vector2();
  window.addEventListener('pointerdown', (e) => {
    if (reduceMotion) return;
    if (toLogoLocal(e.clientX, e.clientY, ripplePoint)) {
      liquid.uRipple.value.set(ripplePoint.x, ripplePoint.y, shared.uTime.value);
      liquid.uRippleAmp.value = 0.035;
    }
  }, { passive: true });

  /* ---- Блики ---- */
  function makeGlintTexture() {
    const size = 256;
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const ctx = c.getContext('2d');
    const mid = size / 2;

    const core = ctx.createRadialGradient(mid, mid, 0, mid, mid, mid * 0.45);
    core.addColorStop(0, 'rgba(255,255,255,1)');
    core.addColorStop(0.2, 'rgba(235,245,255,0.55)');
    core.addColorStop(1, 'rgba(200,225,255,0)');
    ctx.fillStyle = core;
    ctx.fillRect(0, 0, size, size);

    ctx.globalCompositeOperation = 'lighter';
    const ray = (horizontal, length, thickness) => {
      const g = horizontal
        ? ctx.createLinearGradient(mid - length, 0, mid + length, 0)
        : ctx.createLinearGradient(0, mid - length, 0, mid + length);
      g.addColorStop(0, 'rgba(255,255,255,0)');
      g.addColorStop(0.5, 'rgba(255,255,255,0.95)');
      g.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      if (horizontal) ctx.ellipse(mid, mid, length, thickness, 0, 0, Math.PI * 2);
      else ctx.ellipse(mid, mid, thickness, length, 0, 0, Math.PI * 2);
      ctx.fill();
    };
    ray(true, mid * 0.98, 2.2);
    ray(false, mid * 0.98, 2.2);

    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  function setupGlints(geometry) {
    const pos = geometry.attributes.position;
    const nor = geometry.attributes.normal;
    const zMax = geometry.boundingBox.max.z;
    const candidates = [];
    for (let i = 0; i < pos.count; i += 7) {
      if (nor.getZ(i) > 0.55 && nor.getY(i) > 0.25 && pos.getZ(i) > zMax * 0.35) candidates.push(i);
    }
    glintSpots = candidates.map((i) => new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i) + 0.03));
    if (!glintSpots.length) return;

    const tex = makeGlintTexture();
    const count = 3;
    for (let k = 0; k < count; k++) {
      const sprite = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: tex,
          blending: THREE.AdditiveBlending,
          depthTest: false,
          depthWrite: false,
          transparent: true,
          toneMapped: false,
          opacity: 0,
        }),
      );
      sprite.renderOrder = 10;
      logoMesh.add(sprite);
      glints.push({ sprite, start: 1.2 + k * 1.7, life: 1.6, size: 0.42 });
    }
  }

  function respawnGlint(g, now) {
    const spot = glintSpots[(Math.random() * glintSpots.length) | 0];
    g.sprite.position.copy(spot);
    g.start = now + 0.6 + Math.random() * 3.2;
    g.life = 1.2 + Math.random() * 0.9;
    g.size = 0.45 + Math.random() * 0.3;
    g.spin = (Math.random() - 0.5) * 0.6;
  }

  /* ---- Загрузка модели ---- */
  let introStart = -1;

  const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
  loader.load(
    MODEL_URL,
    (gltf) => {
      let src = null;
      gltf.scene.updateMatrixWorld(true);
      gltf.scene.traverse((o) => {
        if (!src && o.isMesh) src = o;
      });
      if (!src) return;

      // Деквантуем в float32 и запекаем трансформ узла — чистое пространство для шейдера
      const srcGeo = src.geometry;
      const count = srcGeo.attributes.position.count;
      const positions = new Float32Array(count * 3);
      const normals = new Float32Array(count * 3);
      const normalMatrix = new THREE.Matrix3().getNormalMatrix(src.matrixWorld);
      const v = new THREE.Vector3();
      for (let i = 0; i < count; i++) {
        v.fromBufferAttribute(srcGeo.attributes.position, i).applyMatrix4(src.matrixWorld);
        positions[i * 3] = v.x;
        positions[i * 3 + 1] = v.y;
        positions[i * 3 + 2] = v.z;
        v.fromBufferAttribute(srcGeo.attributes.normal, i).applyMatrix3(normalMatrix).normalize();
        normals[i * 3] = v.x;
        normals[i * 3 + 1] = v.y;
        normals[i * 3 + 2] = v.z;
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
      geometry.setIndex(srcGeo.index);
      geometry.center();
      geometry.computeBoundingBox();
      geometry.boundingBox.getSize(logoSize);

      // Нормализуем: ширина логотипа = 2 единицы
      const norm = 2 / logoSize.x;
      geometry.scale(norm, norm, norm);
      geometry.computeBoundingBox();
      geometry.computeBoundingSphere();
      geometry.boundingSphere.radius *= 1.2;
      geometry.boundingBox.getSize(logoSize);

      logoMesh = new THREE.Mesh(geometry, material);
      logoMesh.scale.setScalar(0.0001);
      logo.add(logoMesh);
      setupGlints(geometry);
      layout();

      // Прогреваем шейдер до первого кадра, чтобы не было рывка
      renderer.compile(scene, camera);
      introStart = clock.getElapsedTime();
      document.documentElement.classList.add('is-ready');
    },
    undefined,
    (err) => {
      console.error('Не удалось загрузить модель:', err);
      document.documentElement.classList.add('is-ready');
    },
  );

  /* ---- Анимация ---- */
  const easeOutBack = (x) => {
    const c1 = 1.5;
    const c3 = c1 + 1;
    return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
  };
  const motion = reduceMotion ? 0.25 : 1;

  function frame() {
    const dt = Math.min(clock.getDelta(), 0.05);
    const t = clock.getElapsedTime();
    shared.uTime.value = t * (reduceMotion ? 0.35 : 1);

    // Плавное следование за курсором
    const k = 1 - Math.exp(-dt * 3);
    pointer.sx += ((pointer.active ? pointer.x : 0) - pointer.sx) * k;
    pointer.sy += ((pointer.active ? pointer.y : 0) - pointer.sy) * k;
    pointer.energy *= Math.exp(-dt * 1.6);
    skyMat.uniforms.uParallax.value.set(pointer.sx, pointer.sy);

    if (logoMesh) {
      const p = introStart < 0 ? 0 : Math.min(1, (t - introStart) / 1.8);
      const appear = reduceMotion ? p : easeOutBack(p);
      logoMesh.scale.setScalar(Math.max(0.0001, view.scale * (0.55 + 0.45 * appear)));

      // Капля «устаканивается» после появления
      liquid.uAmp.value = (0.014 + (1 - p) * (1 - p) * 0.09) * motion;
      liquid.uPointerAmp.value += (pointer.energy * 0.05 * motion - liquid.uPointerAmp.value) * k;
      liquid.uRippleAmp.value *= Math.exp(-dt * 0.9);

      logo.position.y = view.baseY + Math.sin(t * 0.9) * 0.035 * view.scale * motion;
      logo.rotation.x = (Math.sin(t * 0.55) * 0.06 - pointer.sy * 0.16) * motion;
      logo.rotation.y = (Math.sin(t * 0.42) * 0.1 + pointer.sx * 0.24) * motion;
      logo.rotation.z = Math.sin(t * 0.5) * 0.02 * motion;

      for (const g of glints) {
        if (g.spin === undefined) respawnGlint(g, t);
        const life = (t - g.start) / g.life;
        if (life >= 1) {
          respawnGlint(g, t);
          g.sprite.material.opacity = 0;
          continue;
        }
        const a = life < 0 ? 0 : Math.sin(life * Math.PI);
        g.sprite.material.opacity = a * a * p;
        g.sprite.material.rotation = g.spin * life + 0.15;
        g.sprite.scale.setScalar(g.size * (0.35 + 0.65 * a));
      }
    }

    scene.environmentRotation.y = Math.sin(t * 0.12) * 0.35;

    renderer.setRenderTarget(skyTarget);
    renderer.render(skyScene, orthoCam);
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);
  }

  renderer.setAnimationLoop(frame);
}
