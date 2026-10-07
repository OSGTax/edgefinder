import { CanvasTexture, DynamicDrawUsage, InstancedMesh, Matrix4, MeshBasicMaterial, PlaneGeometry, Vector3, type Scene } from 'three';

// Soft contact shadows: a dark blurred disc under a kid whose real (shadow
// map) shadow is switched off, so far kids and kids on the Fast tier still
// stand on the grass instead of floating. One instanced mesh, one draw call.

function discTexture(): CanvasTexture {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d')!;
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, 'rgba(0,0,0,1)');
  grd.addColorStop(0.45, 'rgba(0,0,0,.7)');
  grd.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, 64, 64);
  return new CanvasTexture(c);
}

export class ContactShadows {
  readonly mesh: InstancedMesh;
  private n = 0;
  private m = new Matrix4();
  private s = new Vector3();
  private p = new Vector3();

  constructor(scene: Scene, max: number) {
    const g = new PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
    const mat = new MeshBasicMaterial({ map: discTexture(), transparent: true, opacity: 0.5, depthWrite: false, color: '#262a1a' });
    mat.name = 'contactShadow';
    this.mesh = new InstancedMesh(g, mat, max);
    this.mesh.instanceMatrix.setUsage(DynamicDrawUsage);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 1;
    this.mesh.name = 'contactShadows';
    this.mesh.count = 0;
    scene.add(this.mesh);
  }

  begin() { this.n = 0; }

  /** A shadow under a kid at `pos` (three space), `size` ft across. */
  add(pos: Vector3, size: number) {
    if (this.n >= this.mesh.instanceMatrix.count) return;
    this.p.set(pos.x, 0.06, pos.z);
    this.s.set(size, 1, size * 0.85);
    this.m.makeScale(this.s.x, 1, this.s.z).setPosition(this.p);
    this.mesh.setMatrixAt(this.n++, this.m);
  }

  end() {
    this.mesh.count = this.n;
    if (this.n) this.mesh.instanceMatrix.needsUpdate = true;
  }
}
