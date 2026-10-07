import {
  AdditiveBlending, BufferAttribute, BufferGeometry, Color, DynamicDrawUsage, Mesh, MeshBasicMaterial, NormalBlending,
  Points, ShaderMaterial, Vector3, type Camera, type Scene, CircleGeometry, CanvasTexture, DoubleSide,
} from 'three';

// Particles (dust, grass bits, splashes, grill smoke, confetti), the ball's
// trail ribbon and its blob shadow. Everything is CPU-simulated and drawn in
// three draw calls.

interface P { p: Vector3; v: Vector3; life: number; age: number; size: number; grow: number; c: Color; a: number; grav: number; drag: number; soft: boolean }

const MAX = 900;

export class Effects {
  private parts: P[] = [];
  private geo = new BufferGeometry();
  private pos = new Float32Array(MAX * 3);
  private col = new Float32Array(MAX * 4);
  private size = new Float32Array(MAX);
  private soft = new Float32Array(MAX);
  readonly points: Points;
  readonly trail: Mesh;
  private trailPts: Vector3[] = [];
  private trailGeo = new BufferGeometry();
  readonly shadow: Mesh;
  /** a thin ring of constant screen size round the ball, so a far ball never vanishes on a phone */
  readonly halo: Points;
  private haloPos = new Float32Array(3);

  constructor(scene: Scene, pixelRatio: number) {
    this.geo.setAttribute('position', new BufferAttribute(this.pos, 3).setUsage(DynamicDrawUsage));
    this.geo.setAttribute('pcol', new BufferAttribute(this.col, 4).setUsage(DynamicDrawUsage));
    this.geo.setAttribute('size', new BufferAttribute(this.size, 1).setUsage(DynamicDrawUsage));
    this.geo.setAttribute('soft', new BufferAttribute(this.soft, 1).setUsage(DynamicDrawUsage));
    const mat = new ShaderMaterial({
      transparent: true, depthWrite: false, blending: NormalBlending,
      uniforms: { uScale: { value: 600 * pixelRatio } },
      vertexShader: /* glsl */ `
        attribute vec4 pcol; attribute float size; attribute float soft;
        varying vec4 vC; varying float vSoft;
        uniform float uScale;
        void main() {
          vC = pcol; vSoft = soft;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * uScale / max(1.0, -mv.z);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */ `
        varying vec4 vC; varying float vSoft;
        void main() {
          vec2 d = gl_PointCoord - 0.5;
          float r = length(d) * 2.0;
          // soft round puffs, or crisp square confetti
          float a = vSoft > 0.5 ? smoothstep(1.0, 0.2, r) : 1.0;
          if (a < 0.01) discard;
          gl_FragColor = vec4(vC.rgb, vC.a * a);
          #include <colorspace_fragment>
        }`,
    });
    this.points = new Points(this.geo, mat);
    this.points.frustumCulled = false;
    this.points.renderOrder = 5;
    scene.add(this.points);

    // trail ribbon
    const n = 24;
    this.trailGeo.setAttribute('position', new BufferAttribute(new Float32Array(n * 2 * 3), 3).setUsage(DynamicDrawUsage));
    this.trailGeo.setAttribute('color', new BufferAttribute(new Float32Array(n * 2 * 4), 4).setUsage(DynamicDrawUsage));
    const idx: number[] = [];
    for (let i = 0; i < n - 1; i++) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 2, a + 1, a + 3); }
    this.trailGeo.setIndex(idx);
    this.trail = new Mesh(this.trailGeo, new MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, blending: AdditiveBlending, side: DoubleSide }));
    this.trail.frustumCulled = false;
    this.trail.renderOrder = 4;
    scene.add(this.trail);

    // soft blob shadow under the ball
    const c = document.createElement('canvas');
    c.width = c.height = 64;
    const g = c.getContext('2d')!;
    const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(0,0,0,0.55)');
    gr.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = gr;
    g.fillRect(0, 0, 64, 64);
    this.shadow = new Mesh(new CircleGeometry(0.5, 20).rotateX(-Math.PI / 2), new MeshBasicMaterial({ map: new CanvasTexture(c), transparent: true, depthWrite: false }));
    this.shadow.renderOrder = 3;
    scene.add(this.shadow);

    const hg = new BufferGeometry();
    hg.setAttribute('position', new BufferAttribute(this.haloPos, 3).setUsage(DynamicDrawUsage));
    this.halo = new Points(hg, new ShaderMaterial({
      transparent: true, depthWrite: false, depthTest: false,
      uniforms: { uSize: { value: 22 * pixelRatio }, uAlpha: { value: 0 } },
      vertexShader: /* glsl */ `
        uniform float uSize;
        void main() {
          gl_PointSize = uSize;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform float uAlpha;
        void main() {
          float r = length(gl_PointCoord - 0.5) * 2.0;
          float ring = smoothstep(0.62, 0.72, r) * (1.0 - smoothstep(0.86, 0.98, r));
          float edge = smoothstep(0.55, 0.62, r) * (1.0 - smoothstep(0.72, 0.8, r));
          vec3 c = mix(vec3(1.0, 0.97, 0.85), vec3(0.17, 0.11, 0.08), edge);
          float a = max(ring, edge * 0.6) * uAlpha;
          if (a < 0.01) discard;
          gl_FragColor = vec4(c, a);
        }`,
    }));
    this.halo.frustumCulled = false;
    this.halo.renderOrder = 12;
    this.halo.visible = false;
    scene.add(this.halo);
  }

  /** show the ring round the ball when it's far from the camera (null hides it) */
  ballHalo(p: Vector3 | null, cam: Camera) {
    const mat = this.halo.material as ShaderMaterial;
    const d = p ? p.distanceTo(cam.position) : 0;
    const want = p ? Math.min(1, Math.max(0, (d - 45) / 40)) : 0;
    mat.uniforms.uAlpha.value += (want - mat.uniforms.uAlpha.value) * 0.25;
    this.halo.visible = mat.uniforms.uAlpha.value > 0.02 && !!p;
    if (p) {
      this.haloPos[0] = p.x; this.haloPos[1] = p.y; this.haloPos[2] = p.z;
      (this.halo.geometry.attributes.position as BufferAttribute).needsUpdate = true;
    }
  }

  private emit(o: Partial<P> & { p: Vector3 }) {
    if (this.parts.length >= MAX) this.parts.shift();
    this.parts.push({
      v: new Vector3(), life: 1, age: 0, size: 0.4, grow: 0, c: new Color('#ffffff'), a: 1, grav: 0, drag: 1.5, soft: true, ...o,
    });
  }

  /** dust kicked up at a three-space point */
  dust(at: Vector3, amount = 1, hex = '#c8a77a') {
    const n = Math.round(6 + amount * 10);
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2, s = (1 + Math.random() * 3) * amount;
      this.emit({
        p: at.clone().add(new Vector3(Math.cos(a) * 0.3, 0.15 + Math.random() * 0.3, Math.sin(a) * 0.3)),
        v: new Vector3(Math.cos(a) * s, 0.8 + Math.random() * 1.5 * amount, Math.sin(a) * s),
        life: 0.7 + Math.random() * 0.8, size: 0.5 + Math.random() * 0.6 * amount, grow: 1.4, c: new Color(hex), a: 0.55, grav: -0.5, drag: 2.5,
      });
    }
  }

  grass(at: Vector3, amount = 1) {
    for (let i = 0; i < 10 * amount; i++) {
      const a = Math.random() * Math.PI * 2, s = 2 + Math.random() * 4;
      this.emit({ p: at.clone().add(new Vector3(0, 0.1, 0)), v: new Vector3(Math.cos(a) * s, 3 + Math.random() * 4, Math.sin(a) * s), life: 0.8, size: 0.09, c: new Color(['#5f9a3a', '#7cb84c', '#4a7f2c'][i % 3]), a: 1, grav: 18, drag: 0.6, soft: false });
    }
  }

  splash(at: Vector3) {
    for (let i = 0; i < 70; i++) {
      const a = Math.random() * Math.PI * 2, s = 2 + Math.random() * 7;
      this.emit({ p: at.clone(), v: new Vector3(Math.cos(a) * s, 9 + Math.random() * 14, Math.sin(a) * s), life: 0.9 + Math.random() * 0.5, size: 0.14 + Math.random() * 0.14, c: new Color(i % 3 ? '#e8f8ff' : '#9fe0f5'), a: 0.9, grav: 32, drag: 0.4 });
    }
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2;
      this.emit({ p: at.clone().add(new Vector3(0, 0.1, 0)), v: new Vector3(Math.cos(a) * 5, 0.5, Math.sin(a) * 5), life: 0.8, size: 0.9, grow: 1.2, c: new Color('#ffffff'), a: 0.5, drag: 3 });
    }
  }

  smoke(at: Vector3) {
    this.emit({
      p: at.clone().add(new Vector3((Math.random() - 0.5) * 0.8, 0, (Math.random() - 0.5) * 0.5)),
      v: new Vector3(0.4 + Math.random() * 0.4, 1.6 + Math.random() * 0.8, -0.2 + Math.random() * 0.3),
      life: 3 + Math.random() * 1.5, size: 0.7, grow: 1.1, c: new Color(Math.random() > 0.5 ? '#d8d8d6' : '#c4c2be'), a: 0.25, drag: 0.2,
    });
  }

  confetti(at: Vector3, n = 160) {
    const cols = ['#ff5e5b', '#ffd23f', '#3fa7ff', '#7dff9a', '#c39bff', '#ffffff', '#ff922b'];
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2, s = 3 + Math.random() * 9;
      this.emit({ p: at.clone(), v: new Vector3(Math.cos(a) * s, 12 + Math.random() * 14, Math.sin(a) * s), life: 2.5 + Math.random() * 2, size: 0.16, c: new Color(cols[i % cols.length]), a: 1, grav: 9, drag: 1.6, soft: false });
    }
  }

  sparkle(at: Vector3, hex = '#ffe14d') {
    for (let i = 0; i < 26; i++) {
      const u = new Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize().multiplyScalar(4 + Math.random() * 5);
      this.emit({ p: at.clone(), v: u, life: 0.6 + Math.random() * 0.4, size: 0.22, c: new Color(hex), a: 1, drag: 3 });
    }
  }

  /** the ball's recent path; call every frame with the ball position (or null to fade out) */
  trailTo(p: Vector3 | null, cam: Camera, strength: number) {
    if (p) {
      const last = this.trailPts[this.trailPts.length - 1];
      if (!last || last.distanceToSquared(p) > 0.04) this.trailPts.push(p.clone());
      if (this.trailPts.length > 24) this.trailPts.shift();
    } else if (this.trailPts.length) this.trailPts.shift();
    const pos = this.trailGeo.attributes.position as BufferAttribute;
    const col = this.trailGeo.attributes.color as BufferAttribute;
    const n = this.trailPts.length;
    const camPos = cam.position;
    for (let i = 0; i < 24; i++) {
      const k = Math.min(i, n - 1);
      if (n < 2) { pos.setXYZ(i * 2, 0, -100, 0); pos.setXYZ(i * 2 + 1, 0, -100, 0); continue; }
      const a = this.trailPts[Math.max(0, k)];
      const b = this.trailPts[Math.min(n - 1, k + 1)] ?? a;
      const tan = b.clone().sub(this.trailPts[Math.max(0, k - 1)]).normalize();
      const side = tan.cross(camPos.clone().sub(a).normalize()).normalize();
      const f = n > 1 ? k / (n - 1) : 0;
      // keep the ribbon a readable width on a small screen however far away it is
      const w = (0.08 * f + 0.01) * Math.max(1, a.distanceTo(camPos) / 45);
      pos.setXYZ(i * 2, a.x + side.x * w, a.y + side.y * w, a.z + side.z * w);
      pos.setXYZ(i * 2 + 1, a.x - side.x * w, a.y - side.y * w, a.z - side.z * w);
      const al = f * f * 0.75 * strength;
      col.setXYZW(i * 2, 1, 0.97, 0.85, al);
      col.setXYZW(i * 2 + 1, 1, 0.97, 0.85, al);
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
  }

  ballShadow(p: Vector3 | null, groundY = 0.04) {
    if (!p) { this.shadow.visible = false; return; }
    this.shadow.visible = true;
    const h = Math.max(0, p.y - groundY);
    const s = 0.7 + h * 0.06;
    this.shadow.scale.setScalar(s);
    this.shadow.position.set(p.x, groundY, p.z);
    (this.shadow.material as MeshBasicMaterial).opacity = Math.max(0.15, 1 - h / 60);
  }

  update(dt: number) {
    let n = 0;
    for (let i = this.parts.length - 1; i >= 0; i--) {
      const q = this.parts[i];
      q.age += dt;
      if (q.age >= q.life) { this.parts.splice(i, 1); continue; }
      q.v.y -= q.grav * dt;
      q.v.multiplyScalar(Math.exp(-q.drag * dt));
      q.p.addScaledVector(q.v, dt);
      if (q.p.y < 0.02 && q.grav > 0) { q.p.y = 0.02; q.v.set(0, 0, 0); }
    }
    for (const q of this.parts) {
      const f = q.age / q.life;
      this.pos[n * 3] = q.p.x; this.pos[n * 3 + 1] = q.p.y; this.pos[n * 3 + 2] = q.p.z;
      this.col[n * 4] = q.c.r; this.col[n * 4 + 1] = q.c.g; this.col[n * 4 + 2] = q.c.b;
      this.col[n * 4 + 3] = q.a * (1 - f) * Math.min(1, q.age * 8);
      this.size[n] = q.size * (1 + q.grow * f);
      this.soft[n] = q.soft ? 1 : 0;
      n++;
    }
    this.geo.setDrawRange(0, n);
    (this.geo.attributes.position as BufferAttribute).needsUpdate = true;
    (this.geo.attributes.pcol as BufferAttribute).needsUpdate = true;
    (this.geo.attributes.size as BufferAttribute).needsUpdate = true;
    (this.geo.attributes.soft as BufferAttribute).needsUpdate = true;
  }
}
