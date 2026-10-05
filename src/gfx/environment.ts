import {
  Color, DirectionalLight, Fog, Group, HemisphereLight, MathUtils, Mesh, MeshBasicMaterial, PMREMGenerator, PlaneGeometry,
  Scene, Vector3, type WebGLRenderer,
} from 'three';
import { cloudTex } from './textures';
import { makeSky, NOON_SKY } from './sky';
import { mulberry } from './noise';
import type { Quality } from './quality';

export interface SunOptions {
  /** degrees above the horizon */
  elevation: number;
  /** degrees, 0 = sun behind home plate, positive toward the first-base side */
  azimuth: number;
  turbidity?: number;
}

/** Sky dome, sun with soft shadows, sky fill light, image-based lighting, clouds and haze. */
export class Environment {
  readonly sky: Mesh;
  readonly sun = new DirectionalLight(0xfff1dc, 2.7);
  readonly hemi = new HemisphereLight(0xc4def5, 0x56683a, 0.6);
  readonly clouds = new Group();
  readonly sunDir = new Vector3();

  constructor(scene: Scene, renderer: WebGLRenderer, q: Quality, o: SunOptions) {
    const phi = MathUtils.degToRad(90 - o.elevation);
    const theta = MathUtils.degToRad(o.azimuth);
    // three: +z is behind home plate, +x toward first base
    this.sunDir.setFromSphericalCoords(1, phi, theta);
    this.sky = makeSky(this.sunDir, NOON_SKY);
    scene.add(this.sky);

    // image-based lighting from the same sky over a green ground bounce
    const pmrem = new PMREMGenerator(renderer);
    const envScene = new Scene();
    envScene.add(makeSky(this.sunDir, NOON_SKY, 500));
    const groundBounce = new Mesh(new PlaneGeometry(2000, 2000), new MeshBasicMaterial({ color: 0x3f6a2a }));
    groundBounce.rotation.x = -Math.PI / 2;
    groundBounce.position.y = -2;
    envScene.add(groundBounce);
    scene.environment = pmrem.fromScene(envScene, 0.02).texture;
    scene.environmentIntensity = 0.6;
    pmrem.dispose();

    // the sun: soft shadows over the whole yard
    const s = this.sun;
    s.castShadow = true;
    s.shadow.mapSize.set(q.shadowMap, q.shadowMap);
    const cam = s.shadow.camera;
    cam.left = -230; cam.right = 230; cam.top = 230; cam.bottom = -230;
    cam.near = 50; cam.far = 1400;
    s.shadow.bias = -0.0004;
    s.shadow.normalBias = 0.45;
    s.shadow.radius = 3;
    s.target.position.set(0, 0, -70);
    s.position.copy(this.sunDir).multiplyScalar(600).add(s.target.position);
    scene.add(s, s.target, this.hemi);

    scene.fog = new Fog(new Color(NOON_SKY.horizon), 900, 6000);

    this.addClouds();
    scene.add(this.clouds);
  }

  private addClouds() {
    const rnd = mulberry(1234);
    for (let i = 0; i < 16; i++) {
      const tex = cloudTex(256, 1 + (i % 5));
      const m = new MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, fog: false, opacity: 0.95 });
      const w = 900 + rnd() * 1300;
      const mesh = new Mesh(new PlaneGeometry(w, w * 0.5), m);
      const a = (rnd() * 1.6 - 0.8) * Math.PI;
      const r = 3200 + rnd() * 1600;
      mesh.position.set(Math.sin(a) * r, 600 + rnd() * 900, -Math.cos(a) * r);
      mesh.lookAt(0, mesh.position.y * 0.6, 0);
      mesh.renderOrder = -1;
      this.clouds.add(mesh);
    }
  }

  update(t: number) {
    // clouds drift slowly across the sky
    this.clouds.rotation.y = t * 0.0015;
  }
}
