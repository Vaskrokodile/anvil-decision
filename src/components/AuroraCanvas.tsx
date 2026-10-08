import { useEffect, useRef } from 'react';
import * as THREE from 'three';

/**
 * Star field over the CSS aurora — ~1400 cool-white points twinkling in the
 * upper sky, plus a sparse ring of brighter points hugging the aurora dome.
 * Cheap shader points; motion is drift + sine twinkle only.
 */

const VERT = /* glsl */ `
  attribute float aSeed;
  uniform float uTime;
  uniform float uPixelRatio;
  varying float vSeed;

  void main() {
    vSeed = aSeed;
    vec3 p = position;
    float t = uTime * 0.06;
    p.x += sin(t + aSeed * 11.0) * 0.35;
    p.y += cos(t * 1.3 + aSeed * 7.0) * 0.28;

    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float size = mix(0.9, 2.2, fract(aSeed * 3.71));
    gl_PointSize = size * uPixelRatio * (24.0 / -mv.z);
  }
`;

const FRAG = /* glsl */ `
  precision mediump float;
  uniform float uTime;
  varying float vSeed;

  void main() {
    vec2 uv = gl_PointCoord - 0.5;
    float d = length(uv);
    float disc = smoothstep(0.5, 0.05, d);
    if (disc < 0.01) discard;

    // cool white → faint blue
    vec3 col = mix(vec3(0.85, 0.9, 1.0), vec3(0.55, 0.7, 1.0), fract(vSeed * 5.13));
    float twinkle = 0.35 + 0.65 * (0.5 + 0.5 * sin(uTime * (0.6 + fract(vSeed) * 1.4) + vSeed * 20.0));
    gl_FragColor = vec4(col, disc * twinkle * 0.8);
  }
`;

export default function AuroraCanvas({ className }: { className?: string }) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    const dpr = Math.min(window.devicePixelRatio, 2);
    renderer.setPixelRatio(dpr);
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      55,
      mount.clientWidth / mount.clientHeight,
      0.1,
      200,
    );
    camera.position.set(0, 0, 26);

    let s = 4242;
    const rng = () => ((s = (s * 16807) % 2147483647) / 2147483647);

    // stars scattered across the sky band (upper 60% of the view)
    const N = 1400;
    const positions = new Float32Array(N * 3);
    const seeds = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      positions[i * 3] = (rng() - 0.5) * 90;
      positions[i * 3 + 1] = 4 + rng() * 16 - Math.abs(rng() - 0.5) * 4;
      positions[i * 3 + 2] = (rng() - 0.5) * 18 - 6;
      seeds[i] = rng() * 100;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));

    const mat = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: { uTime: { value: 0 }, uPixelRatio: { value: dpr } },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    scene.add(new THREE.Points(geo, mat));

    const target = new THREE.Vector2(0, 0);
    const onPointer = (e: PointerEvent) => {
      const r = mount.getBoundingClientRect();
      target.x = ((e.clientX - r.left) / r.width - 0.5) * 2;
      target.y = ((e.clientY - r.top) / r.height - 0.5) * 2;
    };
    mount.addEventListener('pointermove', onPointer);

    const clock = new THREE.Clock();
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const render = () => {
      const t = clock.getElapsedTime();
      mat.uniforms.uTime.value = t;
      camera.position.x += (target.x * 1.6 - camera.position.x) * 0.04;
      camera.position.y += (-target.y * 0.9 - camera.position.y) * 0.04;
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
    };

    if (reduced) render();
    else renderer.setAnimationLoop(render);

    const onResize = () => {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(mount);

    return () => {
      ro.disconnect();
      mount.removeEventListener('pointermove', onPointer);
      renderer.setAnimationLoop(null);
      geo.dispose();
      mat.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  return <div ref={mountRef} className={className} aria-hidden="true" />;
}
