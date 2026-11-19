import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import type { SemanticCluster } from '../data/embeddingsData';

export interface PlotPoint3D {
  id: string;
  label: string;
  sublabel: string;
  cluster: SemanticCluster;
  vector: number[];
  x: number;
  y: number;
  z: number;
  kind: 'static' | 'static-anchor' | 'context-a' | 'context-b' | 'live';
}

interface ProjectedDomLabel {
  id: string;
  label: string;
  screenX: number;
  screenY: number;
  visible: boolean;
  isFocus: boolean;
  isSelected: boolean;
  color: string;
}

interface ProjectedAxisLabel {
  key: string;
  text: string;
  screenX: number;
  screenY: number;
  visible: boolean;
  color: string;
}

interface EmbeddingSpace3DProps {
  points: PlotPoint3D[];
  clusterColors: Record<SemanticCluster, string>;
  explainedVarianceRatio: [number, number, number];
  dataSourceMode: 'reference' | 'live';
  showTrajectories: boolean;
  selectedPair: [string, string];
  onSelectPoint: (pointId: string) => void;
  onHoverPoint: (point: PlotPoint3D | null) => void;
  resetSignal: number;
  autoRotate: boolean;
}

const FLOOR_Y = -3.8;
const WORLD_EXTENT = 3.5;

export function EmbeddingSpace3D({
  points,
  clusterColors,
  explainedVarianceRatio,
  dataSourceMode,
  showTrajectories,
  selectedPair,
  onSelectPoint,
  onHoverPoint,
  resetSignal,
  autoRotate,
}: EmbeddingSpace3DProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [domLabels, setDomLabels] = useState<ProjectedDomLabel[]>([]);
  const [axisLabels, setAxisLabels] = useState<ProjectedAxisLabel[]>([]);
  const [contextLost, setContextLost] = useState<boolean>(false);
  const [webglError, setWebglError] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  // Refs for latest callbacks inside Three.js event listeners
  const onSelectRef = useRef(onSelectPoint);
  const onHoverRef = useRef(onHoverPoint);
  onSelectRef.current = onSelectPoint;
  onHoverRef.current = onHoverPoint;

  // Isotropic world normalization so 3D distances remain faithful to PCA space
  const scaledPoints = useMemo(() => {
    let maxAbs = 0.001;
    for (const p of points) {
      maxAbs = Math.max(maxAbs, Math.abs(p.x), Math.abs(p.y), Math.abs(p.z));
    }
    const scale = WORLD_EXTENT / maxAbs;
    return points.map((p) => ({
      point: p,
      wx: p.x * scale,
      wy: p.y * scale,
      wz: p.z * scale,
    }));
  }, [points]);

  // Persistent scene / camera / controls refs for camera reset & dynamic updates
  const sceneStateRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    dataGroup: THREE.Group;
    pickables: THREE.Mesh[];
    isSoftwareGl: boolean;
    requestRender: () => void;
    disposeDataGroup: () => void;
  } | null>(null);

  // Initialize Three.js WebGL2 Renderer, Studio Lighting & OrbitControls once
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const isSoftwareGl = (() => {
      try {
        const probeCanvas = document.createElement('canvas');
        const gl = probeCanvas.getContext('webgl2');
        if (!gl) return true;
        const ext = gl.getExtension('WEBGL_debug_renderer_info');
        const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
        const gpu = String(gl.getParameter(param));
        gl.getExtension('WEBGL_lose_context')?.loseContext();
        return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
      } catch {
        return true;
      }
    })();

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: !isSoftwareGl,
        powerPreference: 'high-performance',
        alpha: false,
      });
    } catch (err) {
      setWebglError(
        err instanceof Error ? err.message : 'WebGL context unavailable.',
      );
      return;
    }

    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.setPixelRatio(
      isSoftwareGl ? 1 : Math.min(window.devicePixelRatio || 1, 2),
    );
    renderer.setClearColor(0x070b14, 1);

    const rect = container.getBoundingClientRect();
    const initW = Math.max(320, rect.width || 760);
    const initH = Math.max(360, rect.height || 480);
    renderer.setSize(initW, initH);
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.outline = 'none';
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070b14);
    scene.fog = new THREE.FogExp2(0x070b14, 0.032);

    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    const roomEnv = new RoomEnvironment();
    const envTarget = pmremGenerator.fromScene(roomEnv, 0.04);
    scene.environment = envTarget.texture;
    roomEnv.dispose();
    pmremGenerator.dispose();

    const camera = new THREE.PerspectiveCamera(44, initW / initH, 0.1, 100);
    camera.position.set(6.8, 4.6, 7.6);
    camera.lookAt(0, 0, 0);

    // Three-point studio lighting
    const ambientLight = new THREE.AmbientLight(0xcbd5e1, 0.45);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xf8fafc, 1.45);
    keyLight.position.set(6, 9, 6);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x22d3ee, 0.9);
    rimLight.position.set(-7, 4, -6);
    scene.add(rimLight);

    // Static Coordinate Stage (Grid Floor + Calibrated Principal Axes)
    const stageGroup = new THREE.Group();
    const grid = new THREE.GridHelper(9.6, 12, 0x1e293b, 0x111827);
    grid.position.y = FLOOR_Y;
    stageGroup.add(grid);

    const makeAxis = (
      start: THREE.Vector3,
      end: THREE.Vector3,
      hex: number,
    ) => {
      const geo = new THREE.BufferGeometry().setFromPoints([start, end]);
      const mat = new THREE.LineBasicMaterial({
        color: hex,
        transparent: true,
        opacity: 0.65,
      });
      return new THREE.Line(geo, mat);
    };

    stageGroup.add(
      makeAxis(new THREE.Vector3(-4.2, 0, 0), new THREE.Vector3(4.2, 0, 0), 0x22d3ee),
    );
    stageGroup.add(
      makeAxis(new THREE.Vector3(0, FLOOR_Y, 0), new THREE.Vector3(0, 4.1, 0), 0xa78bfa),
    );
    stageGroup.add(
      makeAxis(new THREE.Vector3(0, 0, -4.2), new THREE.Vector3(0, 0, 4.2), 0x60a5fa),
    );
    scene.add(stageGroup);

    const dataGroup = new THREE.Group();
    scene.add(dataGroup);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.06;
    controls.minDistance = 4.2;
    controls.maxDistance = 20;
    controls.maxPolarAngle = Math.PI * 0.85;
    controls.target.set(0, -0.2, 0);
    controls.update();

    let frameId = 0;
    let renderBusy = false;
    let renderDirty = false;
    let disposed = false;

    const afterDraw = (cb: () => void) => {
      const gl = renderer.getContext() as WebGL2RenderingContext;
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const poll = () => {
        if (disposed) return;
        const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
        if (state === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          if (sync) gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
    };

    const projectOverlayLabels = () => {
      const w = renderer.domElement.clientWidth;
      const h = renderer.domElement.clientHeight;
      if (w === 0 || h === 0) return;

      const tmp = new THREE.Vector3();
      const nextDom: ProjectedDomLabel[] = [];

      for (const obj of dataGroup.children) {
        if (!obj.userData?.pointId) continue;
        tmp.copy(obj.position).project(camera);
        const visible = tmp.z >= -1 && tmp.z <= 1;
        const screenX = (tmp.x * 0.5 + 0.5) * w;
        const screenY = (-tmp.y * 0.5 + 0.5) * h;
        nextDom.push({
          id: obj.userData.pointId as string,
          label: obj.userData.label as string,
          screenX,
          screenY,
          visible:
            visible &&
            screenX >= 12 &&
            screenX <= w - 12 &&
            screenY >= 12 &&
            screenY <= h - 12,
          isFocus: Boolean(obj.userData.isFocus),
          isSelected: Boolean(obj.userData.isSelected),
          color: obj.userData.color as string,
        });
      }
      setDomLabels(nextDom);

      // Axis endpoint labels
      const axisEndpoints = [
        {
          key: 'pc1',
          pos: new THREE.Vector3(4.35, 0, 0),
          text: `PC1 (${(explainedVarianceRatio[0] * 100).toFixed(1)}%)`,
          color: '#22D3EE',
        },
        {
          key: 'pc2',
          pos: new THREE.Vector3(0, 4.25, 0),
          text: `PC2 (${(explainedVarianceRatio[1] * 100).toFixed(1)}%)`,
          color: '#A78BFA',
        },
        {
          key: 'pc3',
          pos: new THREE.Vector3(0, 0, 4.35),
          text: `PC3 (${(explainedVarianceRatio[2] * 100).toFixed(1)}%)`,
          color: '#60A5FA',
        },
      ];

      setAxisLabels(
        axisEndpoints.map((ax) => {
          tmp.copy(ax.pos).project(camera);
          const sx = (tmp.x * 0.5 + 0.5) * w;
          const sy = (-tmp.y * 0.5 + 0.5) * h;
          return {
            key: ax.key,
            text: ax.text,
            screenX: sx,
            screenY: sy,
            visible: tmp.z >= -1 && tmp.z <= 1 && sx > 0 && sx < w && sy > 0 && sy < h,
            color: ax.color,
          };
        }),
      );
    };

    const requestRender = () => {
      if (disposed) return;
      if (renderBusy) {
        renderDirty = true;
        return;
      }
      renderBusy = true;
      frameId = requestAnimationFrame(() => {
        if (disposed) return;
        renderDirty = false;
        controls.update();
        renderer.render(scene, camera);
        projectOverlayLabels();
        const done = () => {
          renderBusy = false;
          if (renderDirty || (controls.autoRotate && !isSoftwareGl)) {
            requestRender();
          }
        };
        if (isSoftwareGl) afterDraw(done);
        else done();
      });
    };

    const disposeDataGroup = () => {
      while (dataGroup.children.length > 0) {
        const child = dataGroup.children[0];
        dataGroup.remove(child);
        child.traverse((node) => {
          if ((node as THREE.Mesh).geometry) {
            (node as THREE.Mesh).geometry.dispose();
          }
          const mat = (node as THREE.Mesh).material;
          if (Array.isArray(mat)) {
            mat.forEach((m) => m.dispose());
          } else if (mat) {
            mat.dispose();
          }
        });
      }
    };

    sceneStateRef.current = {
      renderer,
      scene,
      camera,
      controls,
      dataGroup,
      pickables: [],
      isSoftwareGl,
      requestRender,
      disposeDataGroup,
    };

    const onControlsChange = () => requestRender();
    controls.addEventListener('change', onControlsChange);

    // Raycasting for hover and click picking
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let pointerDownPos = { x: 0, y: 0 };

    const getIntersections = (clientX: number, clientY: number) => {
      const canvasRect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((clientX - canvasRect.left) / canvasRect.width) * 2 - 1;
      pointer.y = -((clientY - canvasRect.top) / canvasRect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const state = sceneStateRef.current;
      if (!state) return [];
      return raycaster.intersectObjects(state.pickables, false);
    };

    const onPointerMove = (e: PointerEvent) => {
      const hits = getIntersections(e.clientX, e.clientY);
      const hitPoint = hits[0]?.object?.userData?.plotPoint as PlotPoint3D | undefined;
      if (hitPoint) {
        renderer.domElement.style.cursor = 'pointer';
        setHoveredId(hitPoint.id);
        onHoverRef.current(hitPoint);
      } else {
        renderer.domElement.style.cursor = 'grab';
        setHoveredId(null);
        onHoverRef.current(null);
      }
    };

    const onPointerDown = (e: PointerEvent) => {
      pointerDownPos = { x: e.clientX, y: e.clientY };
    };

    const onPointerUp = (e: PointerEvent) => {
      const dist = Math.hypot(
        e.clientX - pointerDownPos.x,
        e.clientY - pointerDownPos.y,
      );
      if (dist > 6) return; // Was an orbit drag, not a click
      const hits = getIntersections(e.clientX, e.clientY);
      const hitPoint = hits[0]?.object?.userData?.plotPoint as PlotPoint3D | undefined;
      if (hitPoint) {
        onSelectRef.current(hitPoint.id);
      }
    };

    const onContextLost = (e: Event) => {
      e.preventDefault();
      setContextLost(true);
    };
    const onContextRestored = () => {
      setContextLost(false);
      requestRender();
    };

    const domEl = renderer.domElement;
    domEl.addEventListener('pointermove', onPointerMove);
    domEl.addEventListener('pointerdown', onPointerDown);
    domEl.addEventListener('pointerup', onPointerUp);
    domEl.addEventListener('webglcontextlost', onContextLost);
    domEl.addEventListener('webglcontextrestored', onContextRestored);

    const resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry || disposed) return;
      const w = Math.max(320, Math.floor(entry.contentRect.width));
      const h = Math.max(360, Math.floor(entry.contentRect.height));
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      requestRender();
    });
    resizeObserver.observe(container);

    requestRender();

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      resizeObserver.disconnect();
      controls.removeEventListener('change', onControlsChange);
      controls.dispose();
      domEl.removeEventListener('pointermove', onPointerMove);
      domEl.removeEventListener('pointerdown', onPointerDown);
      domEl.removeEventListener('pointerup', onPointerUp);
      domEl.removeEventListener('webglcontextlost', onContextLost);
      domEl.removeEventListener('webglcontextrestored', onContextRestored);

      disposeDataGroup();
      stageGroup.traverse((node) => {
        if ((node as THREE.Line).geometry) (node as THREE.Line).geometry.dispose();
        const mat = (node as THREE.Line).material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else if (mat) mat.dispose();
      });
      envTarget.dispose();
      renderer.dispose();
      if (domEl.parentNode === container) {
        container.removeChild(domEl);
      }
      sceneStateRef.current = null;
    };
  }, []);

  // Sync autoRotate toggle (disabled on software WebGL or reduced motion)
  useEffect(() => {
    const state = sceneStateRef.current;
    if (!state) return;
    const prefersReduced =
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    state.controls.autoRotate =
      autoRotate && !state.isSoftwareGl && !prefersReduced;
    state.controls.autoRotateSpeed = 1.1;
    state.requestRender();
  }, [autoRotate]);

  // Camera reset trigger
  useEffect(() => {
    if (resetSignal === 0) return;
    const state = sceneStateRef.current;
    if (!state) return;
    state.camera.position.set(6.8, 4.6, 7.6);
    state.controls.target.set(0, -0.2, 0);
    state.controls.update();
    state.requestRender();
  }, [resetSignal]);

  // Rebuild 3D scatter nodes, drop stems, shift arrows & measurement lines when data changes
  useEffect(() => {
    const state = sceneStateRef.current;
    if (!state) return;

    state.disposeDataGroup();
    const pickables: THREE.Mesh[] = [];

    const sphereGeo = new THREE.SphereGeometry(0.145, 24, 24);
    const focusGeo = new THREE.OctahedronGeometry(0.24, 0);
    const ringGeo = new THREE.TorusGeometry(0.34, 0.022, 12, 36);
    const selectHaloGeo = new THREE.TorusGeometry(0.42, 0.024, 12, 40);

    const posMap = new Map<string, THREE.Vector3>();

    for (const item of scaledPoints) {
      const { point, wx, wy, wz } = item;
      const pos = new THREE.Vector3(wx, wy, wz);
      posMap.set(point.id, pos);

      const isFocus = point.kind !== 'static';
      const isSelected = selectedPair.includes(point.id);
      const isHovered = hoveredId === point.id;
      const colorHex = clusterColors[point.cluster];

      // Composite node group
      const nodeGroup = new THREE.Group();
      nodeGroup.position.copy(pos);
      nodeGroup.userData = {
        pointId: point.id,
        label: point.label,
        isFocus,
        isSelected,
        color: colorHex,
      };

      const coreMat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(colorHex),
        emissive: new THREE.Color(colorHex),
        emissiveIntensity: isSelected || isHovered ? 0.55 : isFocus ? 0.35 : 0.14,
        roughness: isFocus ? 0.2 : 0.35,
        metalness: isFocus ? 0.45 : 0.2,
      });

      const coreMesh = new THREE.Mesh(
        isFocus ? focusGeo.clone() : sphereGeo.clone(),
        coreMat,
      );
      if (isHovered || isSelected) {
        coreMesh.scale.setScalar(1.22);
      }
      coreMesh.userData = { plotPoint: point };
      nodeGroup.add(coreMesh);
      pickables.push(coreMesh);

      if (isFocus) {
        const ringMat = new THREE.MeshStandardMaterial({
          color: 0xf8fafc,
          roughness: 0.25,
          metalness: 0.75,
        });
        const ringMesh = new THREE.Mesh(ringGeo.clone(), ringMat);
        ringMesh.rotation.x = Math.PI * 0.38;
        ringMesh.rotation.y = Math.PI * 0.2;
        nodeGroup.add(ringMesh);
      }

      if (isSelected) {
        const haloMat = new THREE.MeshBasicMaterial({
          color: 0xffffff,
          transparent: true,
          opacity: 0.85,
        });
        const haloMesh = new THREE.Mesh(selectHaloGeo.clone(), haloMat);
        haloMesh.rotation.x = Math.PI / 2;
        nodeGroup.add(haloMesh);
      }

      state.dataGroup.add(nodeGroup);

      // Subtle vertical projection stem to floor grid for 3D depth reading
      const stemGeo = new THREE.BufferGeometry().setFromPoints([
        pos,
        new THREE.Vector3(wx, FLOOR_Y, wz),
      ]);
      const stemMat = new THREE.LineBasicMaterial({
        color: new THREE.Color(colorHex),
        transparent: true,
        opacity: isFocus ? 0.45 : 0.18,
      });
      state.dataGroup.add(new THREE.Line(stemGeo, stemMat));
    }

    sphereGeo.dispose();
    focusGeo.dispose();
    ringGeo.dispose();
    selectHaloGeo.dispose();

    // 3D Contextualization Trajectory Arrows (Static Anchor -> Context A / Context B)
    if (dataSourceMode === 'reference' && showTrajectories) {
      const anchorPos = posMap.get('static-anchor');
      const ctxAPos = posMap.get('ctx-a');
      const ctxBPos = posMap.get('ctx-b');

      const addTrajectory = (
        from: THREE.Vector3,
        to: THREE.Vector3,
        color: number,
      ) => {
        const dir = new THREE.Vector3().subVectors(to, from);
        const len = dir.length();
        if (len < 0.2) return;
        const unit = dir.clone().normalize();
        const arrowHelper = new THREE.ArrowHelper(
          unit,
          from,
          Math.max(0.1, len - 0.25),
          color,
          0.26,
          0.13,
        );
        state.dataGroup.add(arrowHelper);
      };

      if (anchorPos && ctxAPos) addTrajectory(anchorPos, ctxAPos, 0x10b981);
      if (anchorPos && ctxBPos) addTrajectory(anchorPos, ctxBPos, 0x06b6d4);
    }

    // Selected pair distance measurement line in 3D space
    const p1Pos = posMap.get(selectedPair[0]);
    const p2Pos = posMap.get(selectedPair[1]);
    if (p1Pos && p2Pos) {
      const lineGeo = new THREE.BufferGeometry().setFromPoints([p1Pos, p2Pos]);
      const lineMat = new THREE.LineDashedMaterial({
        color: 0xf8fafc,
        dashSize: 0.18,
        gapSize: 0.12,
        transparent: true,
        opacity: 0.72,
      });
      const pairLine = new THREE.Line(lineGeo, lineMat);
      pairLine.computeLineDistances();
      state.dataGroup.add(pairLine);
    }

    state.pickables = pickables;
    state.requestRender();
  }, [
    scaledPoints,
    clusterColors,
    dataSourceMode,
    showTrajectories,
    selectedPair,
    hoveredId,
  ]);

  if (webglError || contextLost) {
    return (
      <div className="flex h-[480px] w-full flex-col items-center justify-center rounded-lg border border-slate-800 bg-[#070B14] p-6 text-center">
        <p className="text-sm font-medium text-slate-200">
          3D WebGL viewport could not be initialized
        </p>
        <p className="mt-1 text-xs text-slate-400">
          {webglError ?? 'WebGL context was lost. Try resetting the view.'}
        </p>
      </div>
    );
  }

  return (
    <div className="relative h-[490px] w-full overflow-hidden rounded-lg border border-slate-800 bg-[#070B14]">
      <div
        ref={containerRef}
        className="h-full w-full"
        role="region"
        aria-label="Interactive 3D PCA embedding space. Drag to orbit, scroll to zoom, click any two points to compare."
      />

      {/* Semantic DOM Axis Labels */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden z-10">
        {axisLabels.map(
          (ax) =>
            ax.visible && (
              <span
                key={ax.key}
                className="absolute -translate-x-1/2 -translate-y-1/2 rounded bg-[#070B14]/80 px-1.5 py-0.5 font-mono text-[10px] font-medium whitespace-nowrap"
                style={{
                  left: ax.screenX,
                  top: ax.screenY,
                  color: ax.color,
                }}
              >
                {ax.text}
              </span>
            ),
        )}

        {/* Projected Node Labels */}
        {domLabels.map(
          (lbl) =>
            lbl.visible && (
              <span
                key={lbl.id}
                className={`absolute font-mono whitespace-nowrap transition-opacity duration-75 ${
                  lbl.isFocus || lbl.isSelected
                    ? 'text-[11px] font-semibold text-slate-50 bg-[#070B14]/80 px-1.5 py-0.5 rounded border border-slate-700/70'
                    : 'text-[10px] text-slate-300/85'
                }`}
                style={{
                  left: lbl.screenX + 10,
                  top: lbl.screenY - 8,
                }}
              >
                {lbl.label}
              </span>
            ),
        )}
      </div>

      {/* Subtle Corner Interaction Hint */}
      <div className="pointer-events-none absolute bottom-2.5 left-3 z-10 rounded bg-[#070B14]/75 px-2.5 py-1 font-mono text-[10px] text-slate-400 border border-slate-800/80">
        Drag to orbit 360° · Scroll to zoom · Right-drag to pan · Click node to
        measure
      </div>
    </div>
  );
}
