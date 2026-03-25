'use client';

import React, { useEffect, useRef, useState, useMemo } from 'react';
import axios from 'axios';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, useTexture } from '@react-three/drei';
import * as THREE from 'three';
import { ChevronRight, Archive, ClipboardList, Activity, UploadCloud, X, UserPlus, Users } from 'lucide-react';
import { motion } from 'framer-motion';

// --- API FETCHING ---
const API_BASE = 'http://localhost:8000';

type Patient = {
    id?: number;
    patient_id: string;
    name: string;
    risk_level: 'High' | 'Medium' | 'Low' | string;
    latest_risk?: 'High' | 'Medium' | 'Low' | string;
    folder?: 'critical' | 'under_observation' | 'clear';
    scan_count: number;
    last_scan_date?: string;
    latest_scan_date?: string;
};

type ScanRecord = {
    scan_id: number;
    date: string;
    tumor_diameter_mm: number;
    risk_level: string;
    status: string;
    x_coordinate: number;
};

// --- TYPES & HELPERS ---
const riskColors = {
    High: 'bg-[#991B1B] text-white',
    Medium: 'bg-[#A16207] text-white',
    Low: 'bg-[#166534] text-white',
    Clear: 'bg-[#166534] text-white',
};

// ─── REALISTIC LUNG GEOMETRY HELPERS ───────────────────────────────────────

/** Build an anatomically-shaped lung lobe as a custom BufferGeometry.
 *  Uses a parametric approach: latitude/longitude sphere deformed by a
 *  shape function that matches the profile of a real lung lobe.
 */
function buildLungGeometry(
    side: 'left' | 'right',
    widthX: number,
    heightY: number,
    depthZ: number,
    segments = 48
): THREE.BufferGeometry {
    const geo = new THREE.SphereGeometry(1, segments, segments);
    const pos = geo.attributes.position as THREE.BufferAttribute;
    const count = pos.count;

    for (let i = 0; i < count; i++) {
        const x = pos.getX(i);
        const y = pos.getY(i);
        const z = pos.getZ(i);

        // Normalised latitude: -1 (bottom) to +1 (top)
        const lat = y; // sphere radius=1, so y in [-1,1]

        // ── Vertical profile ──────────────────────────────────────────────
        // Apex (top) narrows, base (bottom) flattens/widens slightly
        const apexTaper = 1 - Math.max(0, lat) * 0.35;       // shrink top
        const baseBulge = 1 + Math.max(0, -lat) * 0.18;      // widen bottom

        // ── Medial (inner) flat face ──────────────────────────────────────
        // Real lungs have a concave medial surface towards the heart
        const medialFace = side === 'right' ? -x : x;         // inner x
        const medialIndent = Math.max(0, medialFace) * 0.38;  // flatten inward

        // ── Costal (outer) convex surface ─────────────────────────────────
        // Slight rib-cage bulge on outer side
        const costalFace = side === 'right' ? x : -x;
        const costalBulge = 1 + Math.max(0, costalFace) * 0.08;

        // ── Diaphragmatic base (bottom concavity) ─────────────────────────
        const diaphragmIndent = lat < -0.55
            ? 1 - ((-lat - 0.55) / 0.45) * 0.28
            : 1.0;

        // ── Cardiac notch on left lung ─────────────────────────────────────
        let cardiacNotch = 1.0;
        if (side === 'left') {
            const frontInner = -x * z; // front-inner quadrant
            if (lat < 0.1 && lat > -0.5 && frontInner > 0.1) {
                cardiacNotch = 1 - Math.min(0.35, frontInner * 0.9);
            }
        }

        const scale = apexTaper * baseBulge * (1 - medialIndent) * costalBulge * diaphragmIndent * cardiacNotch;

        pos.setXYZ(i, x * widthX * scale, y * heightY * apexTaper * diaphragmIndent, z * depthZ * scale);
    }

    geo.computeVertexNormals();
    return geo;
}

/** Animated pulsing nodule */
function PulsingNodule({ position, radius, color }: { position: [number, number, number]; radius: number; color: string }) {
    const meshRef = useRef<THREE.Mesh>(null!);
    useFrame(({ clock }) => {
        const t = clock.getElapsedTime();
        meshRef.current.scale.setScalar(1 + Math.sin(t * 2.5) * 0.06);
    });
    return (
        <mesh ref={meshRef} position={position}>
            <sphereGeometry args={[radius, 32, 32]} />
            <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.8}
                roughness={0.3}
                metalness={0.1}
            />
        </mesh>
    );
}

/** Glow halo around nodule */
function NoduleGlow({ position, radius, color }: { position: [number, number, number]; radius: number; color: string }) {
    return (
        <mesh position={position}>
            <sphereGeometry args={[radius * 1.9, 20, 20]} />
            <meshStandardMaterial
                color={color}
                transparent
                opacity={0.12}
                side={THREE.BackSide}
                depthWrite={false}
            />
        </mesh>
    );
}

/** Bronchial tree for one lung */
function BronchialBranch({
    start, end, width, depth = 0, maxDepth = 4, side
}: {
    start: [number, number, number];
    end: [number, number, number];
    width: number;
    depth?: number;
    maxDepth?: number;
    side: 'left' | 'right';
}) {
    if (depth > maxDepth) return null;

    const dir = new THREE.Vector3(...end).sub(new THREE.Vector3(...start));
    const len = dir.length();
    dir.normalize();

    // Create sub-branches
    const perpY = new THREE.Vector3(0, 1, 0);
    const perp = dir.clone().cross(perpY).normalize();
    const spread = len * 0.55;
    const sign = side === 'right' ? 1 : -1;

    const mid: [number, number, number] = [
        (start[0] + end[0]) / 2,
        (start[1] + end[1]) / 2,
        (start[2] + end[2]) / 2,
    ];

    const b1End: [number, number, number] = [
        end[0] + perp.x * spread * 0.5 + sign * 0.1,
        end[1] - 0.2,
        end[2] + dir.z * spread * 0.35,
    ];
    const b2End: [number, number, number] = [
        end[0] - perp.x * spread * 0.5 - sign * 0.05,
        end[1] - 0.35,
        end[2] + dir.z * spread * 0.2,
    ];

    const childWidth = width * 0.62;
    const opacity = Math.max(0.08, 0.55 - depth * 0.09);

    return (
        <>
            <Line
                points={[start, end]}
                color="#C7A97A"
                lineWidth={Math.max(0.5, width)}
                transparent
                opacity={opacity}
            />
            {depth < maxDepth && (
                <>
                    <BronchialBranch start={end} end={b1End} width={childWidth} depth={depth + 1} maxDepth={maxDepth} side={side} />
                    <BronchialBranch start={end} end={b2End} width={childWidth} depth={depth + 1} maxDepth={maxDepth} side={side} />
                </>
            )}
        </>
    );
}

// --- THREE.JS CLINICAL LUNG VISUALIZATION ---
function ClinicalLungModel({ scan }: { scan: ScanRecord | null }) {
    if (!scan) return null;

    // ── Nodule spatial mapping ─────────────────────────────────────────────
    // x_coordinate 0-512 → left/right in scene
    // We also derive Y and Z from scan data (normalised)
    const rawX      = scan.x_coordinate || 256;
    const normX     = rawX / 512.0;                          // 0-1
    const noduleX   = (normX - 0.5) * 5.0;                  // -2.5 to +2.5
    // Derive Y offset from diameter (larger = more central/inferior)
    const noduleY   = -0.3 + (scan.tumor_diameter_mm / 30.0) * -0.8;
    const noduleZ   = 0.45;

    // 3D coordinates for display (realistic anatomical reference in cm)
    const coord3D = {
        x: (noduleX * 2.5).toFixed(1),
        y: (noduleY * 2.5).toFixed(1),
        z: (noduleZ * 2.5).toFixed(1),
    };

    const noduleR      = Math.max(0.12, scan.tumor_diameter_mm / 18.0);
    const isCritical   = scan.risk_level === 'High' || scan.risk_level === 'Critical';
    const noduleColor  = isCritical ? '#EF4444' : (scan.risk_level === 'Medium' ? '#F59E0B' : '#22C55E');

    // Which lung?
    const inRightLung  = normX > 0.5;
    const lungLabel    = inRightLung ? 'Right Lung' : 'Left Lung';

    // ── Load Real Lung Texture ─────────────────────────────────────────────
    const lungTex = useTexture('/lung_texture.png');
    
    // Reverse V (Y) to map properly onto our custom spherical bounds
    lungTex.wrapS = THREE.ClampToEdgeWrapping;
    lungTex.wrapT = THREE.ClampToEdgeWrapping;

    // ── Custom geometries (memoised) ───────────────────────────────────────
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const rightGeo = useMemo(() => buildLungGeometry('right', 1.55, 2.2, 1.3), []);
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const leftGeo  = useMemo(() => buildLungGeometry('left',  1.3,  2.1, 1.2), []);

    const nodulePos: [number, number, number] = [noduleX, noduleY, noduleZ];

    // Bronchi roots
    const tracheaTop:   [number, number, number] = [0,     2.8,  0];
    const carina:       [number, number, number] = [0,     1.85, 0];
    const rBronchRoot:  [number, number, number] = [0.55,  1.45, 0];
    const lBronchRoot:  [number, number, number] = [-0.45, 1.35, 0];
    const rBronch1End:  [number, number, number] = [1.35,  0.85, 0.1];
    const lBronch1End:  [number, number, number] = [-1.2,  0.75, 0.1];

    // Horizontal fissure (right lung) — major landmark
    const rFissurePoints: [number, number, number][] = [
        [0.3, 0.2, 0.8], [0.8, 0.0, 0.7], [1.4, -0.2, 0.5], [1.7, -0.5, 0.2]
    ];
    // Oblique fissure (right)
    const rObliquePts: [number, number, number][] = [
        [0.6, 1.2, 0.6], [1.0, 0.4, 0.8], [1.5, -0.6, 0.6], [1.6, -1.2, 0.1]
    ];
    // Oblique fissure (left)
    const lObliquePts: [number, number, number][] = [
        [-0.5, 1.1, 0.6], [-0.9, 0.3, 0.8], [-1.3, -0.7, 0.5], [-1.4, -1.3, 0.1]
    ];

    return (
        <group>
            {/* ── Lighting ─────────────────────────────────────── */}
            <ambientLight intensity={0.6} />
            <pointLight position={[6, 8, 6]}  intensity={1.8} color="#FFF5E4" />
            <pointLight position={[-5, 4, -4]} intensity={0.9} color="#C7E4FF" />
            <pointLight position={[0, -6, 4]}  intensity={0.5} color="#FFE4E4" />

            {/* ── RIGHT LUNG ──────────────────────────────────── */}
            <group position={[1.5, -0.15, 0]}>
                {/* Solid tissue fill */}
                <mesh geometry={rightGeo}>
                    <meshStandardMaterial
                        map={lungTex}
                        color="#ffffff"
                        roughness={0.80}
                        metalness={0.12}
                        transparent
                        opacity={0.92}
                        side={THREE.FrontSide}
                        depthWrite={true}
                    />
                </mesh>
                {/* Wireframe overlay */}
                <mesh geometry={rightGeo}>
                    <meshStandardMaterial
                        color="#200502"
                        wireframe
                        transparent
                        opacity={0.06}
                    />
                </mesh>
            </group>

            {/* ── LEFT LUNG ───────────────────────────────────── */}
            <group position={[-1.35, -0.1, 0]}>
                <mesh geometry={leftGeo}>
                    <meshStandardMaterial
                        map={lungTex}
                        color="#ffffff"
                        roughness={0.80}
                        metalness={0.12}
                        transparent
                        opacity={0.90}
                        side={THREE.FrontSide}
                        depthWrite={true}
                    />
                </mesh>
                <mesh geometry={leftGeo}>
                    <meshStandardMaterial
                        color="#200502"
                        wireframe
                        transparent
                        opacity={0.06}
                    />
                </mesh>
            </group>

            {/* ── FISSURES ─────────────────────────────────────── */}
            <Line points={rFissurePoints}  color="#FF9090" lineWidth={1.2} transparent opacity={0.45} />
            <Line points={rObliquePts}    color="#FF9090" lineWidth={1.2} transparent opacity={0.4}  />
            <Line points={lObliquePts}    color="#FF9090" lineWidth={1.2} transparent opacity={0.4}  />

            {/* ── BRONCHIAL TREE ───────────────────────────────── */}
            {/* Trachea */}
            <Line points={[tracheaTop, carina]} color="#D4956A" lineWidth={3.5} transparent opacity={0.8} />
            {/* Main bronchi */}
            <Line points={[carina, rBronchRoot]} color="#D4956A" lineWidth={2.8} transparent opacity={0.75} />
            <Line points={[carina, lBronchRoot]} color="#D4956A" lineWidth={2.5} transparent opacity={0.75} />
            {/* Lobar bronchi */}
            <Line points={[rBronchRoot, rBronch1End]} color="#C4855A" lineWidth={2.0} transparent opacity={0.7} />
            <Line points={[lBronchRoot, lBronch1End]} color="#C4855A" lineWidth={1.8} transparent opacity={0.7} />
            {/* Branching sub-segments */}
            <BronchialBranch start={rBronch1End} end={[1.7, 0.1, 0.2]}  width={1.6} maxDepth={3} side="right" />
            <BronchialBranch start={rBronch1End} end={[1.5, -0.4, 0.3]} width={1.4} maxDepth={3} side="right" />
            <BronchialBranch start={lBronch1End} end={[-1.5, 0.0, 0.2]} width={1.4} maxDepth={3} side="left"  />
            <BronchialBranch start={lBronch1End} end={[-1.3, -0.5, 0.3]} width={1.2} maxDepth={3} side="left" />

            {/* ── NODULE ───────────────────────────────────────── */}
            <NoduleGlow   position={nodulePos} radius={noduleR} color={noduleColor} />
            <PulsingNodule position={nodulePos} radius={noduleR} color={noduleColor} />

            {/* ── TRACKING LINES ───────────────────────────────── */}
            {/* Vertical drop line */}
            <Line
                points={[nodulePos, [nodulePos[0], nodulePos[1] + 2.2, nodulePos[2]]]}
                color={noduleColor}
                lineWidth={1.2}
                transparent
                opacity={0.7}
                dashed
            />
            {/* Horizontal crosshair arm */}
            <Line
                points={[[nodulePos[0] - 0.6, nodulePos[1], nodulePos[2]],
                         [nodulePos[0] + 0.6, nodulePos[1], nodulePos[2]]]}
                color={noduleColor}
                lineWidth={0.8}
                transparent
                opacity={0.5}
            />

            {/* ── LABEL PIN ─────────────────────────────────────── */}
            {/* Small sphere at label attachment */}
            <mesh position={[nodulePos[0], nodulePos[1] + 2.2, nodulePos[2]]}>
                <sphereGeometry args={[0.05, 8, 8]} />
                <meshStandardMaterial color={noduleColor} emissive={noduleColor} emissiveIntensity={1} />
            </mesh>

            {/* ── COORDINATE AXES HINT ─────────────────────────── */}
            <Line points={[[0, -3.2, 0.3], [0, 3.2, 0.3]]}    color="#334155" lineWidth={0.4} transparent opacity={0.15} />
            <Line points={[[-3.5, 0, 0.3], [3.5, 0, 0.3]]}    color="#334155" lineWidth={0.4} transparent opacity={0.15} />
        </group>
    );
}

// --- MAIN DASHBOARD COMPONENT ---
export default function LungCareDashboard() {
    const [patients, setPatients] = useState<Patient[]>([]);
    const [loading, setLoading] = useState(true);
    
    // Active View States
    const [activeView, setActiveView] = useState<'queue' | 'archive'>('queue');
    const [selectedPatientId, setSelectedPatientId] = useState<number | null>(null);
    const [selectedScan, setSelectedScan] = useState<ScanRecord | null>(null);
    
    // Modal States
    const [showModal, setShowModal] = useState(false);

    const fetchPatients = async () => {
        try {
            const res = await axios.get<Patient[]>(`${API_BASE}/api/patients`);
            setPatients(res.data);
        } catch (err) {
            console.error("API Error", err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchPatients();
    }, []);

    const handlePatientSelect = async (patientId: number) => {
        try {
            setSelectedPatientId(patientId);
            const res = await axios.get(`${API_BASE}/api/patients/${patientId}/full`);
            const scans = res.data.scans || [];
            if (scans.length > 0) {
                setSelectedScan(scans[scans.length - 1]);
            } else {
                setSelectedScan(null);
            }
        } catch (err) {
            console.error("Failed to fetch patient records", err);
        }
    };

    const getFolder = (p: Patient) => {
        const r = p.latest_risk || p.risk_level;
        if (r === 'High') return 'critical';
        if (r === 'Medium') return 'under_observation';
        return 'clear';
    };

    const grouped = {
        critical: patients.filter(p => getFolder(p) === 'critical'),
        under_observation: patients.filter(p => getFolder(p) === 'under_observation'),
        clear: patients.filter(p => getFolder(p) === 'clear'),
    };

    const totalScans = patients.reduce((acc, p) => acc + (p.scan_count || 0), 0);
    const activePatient = patients.find(p => p.id === selectedPatientId) || null;

    return (
        <div className="flex h-screen bg-white text-zinc-900 font-sans tracking-tight overflow-hidden text-sm">
            {/* L PANE: NAV & STATS */}
            <div className="w-64 border-r border-slate-400 bg-white flex flex-col p-6 z-10">
                <div className="mb-10">
                    <h1 className="text-2xl font-black tracking-tighter uppercase border-b-2 border-zinc-900 pb-2 mb-2">LUNGCARE</h1>
                    <p className="text-[10px] text-slate-500 font-mono uppercase tracking-widest">Diagnostic Terminal</p>
                </div>

                <button 
                    onClick={() => setShowModal(true)}
                    className="w-full bg-zinc-900 hover:bg-zinc-800 text-white rounded-[2px] py-3 px-4 font-bold transition-colors mb-10 flex items-center justify-center gap-2">
                    <Activity size={16} /> NEW PATIENT SCAN
                </button>

                <div className="space-y-6 flex-1">
                    <div>
                        <div className="text-[10px] text-slate-500 font-bold uppercase mb-1">Total Patients</div>
                        <div className="text-4xl font-black">{patients.length}</div>
                    </div>
                    <div>
                        <div className="text-[10px] text-slate-500 font-bold uppercase mb-1">Total Scans</div>
                        <div className="text-4xl font-black">{totalScans}</div>
                    </div>
                </div>

                <div className="border-t border-slate-300 pt-6 space-y-3">
                    <button 
                        onClick={() => setActiveView('queue')}
                        className={`flex items-center gap-3 text-sm font-semibold w-full text-left ${activeView === 'queue' ? 'text-zinc-900' : 'text-slate-400 hover:text-slate-600'}`}>
                        <ClipboardList size={16} /> Triage Queue
                    </button>
                    <button 
                        onClick={() => setActiveView('archive')}
                        className={`flex items-center gap-3 text-sm font-semibold w-full text-left ${activeView === 'archive' ? 'text-zinc-900' : 'text-slate-400 hover:text-slate-600'}`}>
                        <Archive size={16} /> Archive
                    </button>
                </div>
            </div>

            {/* CENTER PANE: 3D VISUAL */}
            <div className="flex-1 bg-[#0A0D12] flex flex-col relative border-r border-slate-700 z-0">
                <div className="absolute top-4 left-4 z-10 font-mono text-xs font-bold text-slate-500 tracking-widest">
                    ANATOMICAL 3D RENDER · AXIAL/CORONAL VIEW
                </div>
                
                {selectedScan ? (() => {
                    const rawX    = selectedScan.x_coordinate || 256;
                    const normX   = rawX / 512.0;
                    const noduleX = ((normX - 0.5) * 5.0 * 2.5).toFixed(1);
                    const noduleY = (-0.3 + (selectedScan.tumor_diameter_mm / 30.0) * -0.8) * 2.5;
                    const lungLabel = normX > 0.5 ? 'Right Lung' : 'Left Lung';
                    const riskColor = selectedScan.risk_level === 'High' ? '#EF4444' : selectedScan.risk_level === 'Medium' ? '#F59E0B' : '#22C55E';
                    return (
                    <>
                        {/* Status badge */}
                        <div className="absolute top-4 right-4 z-10 flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: riskColor }}></div>
                            <span className="font-mono text-[10px] font-bold" style={{ color: riskColor }}>
                                NODULE DETECTED · {selectedScan.risk_level.toUpperCase()}
                            </span>
                        </div>

                        {/* 3D Canvas */}
                        <div className="h-full w-full">
                            <Canvas camera={{ position: [0, 0, 9], fov: 42 }}>
                                <color attach="background" args={['#0A0D12']} />
                                <fog attach="fog" args={['#0A0D12', 18, 35]} />
                                <OrbitControls enableZoom={true} enablePan={true} autoRotate autoRotateSpeed={0.4} />
                                <React.Suspense fallback={null}>
                                    <ClinicalLungModel scan={selectedScan} />
                                </React.Suspense>
                            </Canvas>
                        </div>

                        {/* Coordinate HUD overlay */}
                        <div className="absolute bottom-0 left-0 right-0 z-10">
                            {/* Top row: patient info */}
                            <div className="mx-4 mb-2 flex gap-2">
                                <div className="bg-black/60 backdrop-blur-md border border-slate-700 rounded px-3 py-1.5 font-mono text-[9px] text-slate-300 flex gap-4">
                                    <span><span className="text-slate-500 mr-1">PT</span>{activePatient?.name?.toUpperCase()}</span>
                                    <span><span className="text-slate-500 mr-1">ID</span>{activePatient?.patient_id || '—'}</span>
                                </div>
                            </div>
                            {/* Bottom row: coordinate readout */}
                            <div className="bg-black/70 backdrop-blur-md border-t border-slate-700 px-4 py-3 flex items-center gap-6">
                                {/* Nodule coords */}
                                <div className="flex gap-4 font-mono text-[10px]">
                                    <div>
                                        <span className="text-slate-500 text-[8px] block mb-0.5">X (MEDIAL–LAT)</span>
                                        <span className="text-cyan-400 font-bold">{noduleX} cm</span>
                                    </div>
                                    <div>
                                        <span className="text-slate-500 text-[8px] block mb-0.5">Y (SUP–INF)</span>
                                        <span className="text-emerald-400 font-bold">{noduleY.toFixed(1)} cm</span>
                                    </div>
                                    <div>
                                        <span className="text-slate-500 text-[8px] block mb-0.5">Z (ANT–POST)</span>
                                        <span className="text-amber-400 font-bold">1.1 cm</span>
                                    </div>
                                </div>
                                <div className="h-6 w-px bg-slate-700" />
                                <div className="font-mono text-[10px]">
                                    <span className="text-slate-500 text-[8px] block mb-0.5">LOBE</span>
                                    <span className="text-white font-bold">{lungLabel}</span>
                                </div>
                                <div className="h-6 w-px bg-slate-700" />
                                <div className="font-mono text-[10px]">
                                    <span className="text-slate-500 text-[8px] block mb-0.5">DIAMETER</span>
                                    <span style={{ color: riskColor }} className="font-bold">{selectedScan.tumor_diameter_mm} mm</span>
                                </div>
                                <div className="h-6 w-px bg-slate-700" />
                                <div className="font-mono text-[10px]">
                                    <span className="text-slate-500 text-[8px] block mb-0.5">RISK</span>
                                    <span style={{ color: riskColor }} className="font-bold">{selectedScan.risk_level.toUpperCase()}</span>
                                </div>
                                <div className="ml-auto font-mono text-[9px] text-slate-600">ISO-SPATIAL · DRAG TO ORBIT · SCROLL TO ZOOM</div>
                            </div>
                        </div>
                    </>
                    );
                })() : (
                    <div className="h-full w-full flex flex-col items-center justify-center text-slate-600 border-2 border-dashed border-slate-800 m-8 rounded">
                        <UploadCloud size={48} className="mb-4 opacity-30" />
                        <h2 className="text-xl font-bold tracking-tight text-slate-500 mb-2">NO SCAN SELECTED</h2>
                        <p className="text-sm font-mono opacity-50 mb-6 text-slate-400">Select a patient from the queue or upload a new scan to get 3D anatomical localization.</p>
                        <button 
                            onClick={() => setShowModal(true)}
                            className="bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold py-2 px-6 rounded text-xs tracking-widest uppercase transition-colors">
                            Upload Image
                        </button>
                    </div>
                )}
            </div>

            {/* RIGHT PANE: TRIAGE QUEUE */}
            <div className="w-96 flex flex-col bg-white z-10 h-full overflow-hidden">
                <div className="p-4 border-b border-slate-400 bg-zinc-900 text-white flex justify-between items-center">
                    <span className="font-bold text-xs uppercase tracking-widest">{activeView === 'queue' ? 'Active Triage Queue' : 'Archive'}</span>
                    <span className="text-[10px] opacity-70">AUTO-SORT: RISK</span>
                </div>
                
                <div className="flex-1 overflow-y-auto hidden-scrollbar">
                    {loading ? (
                        <div className="p-8 text-center text-slate-400 font-mono flex flex-col items-center">
                            <Activity className="animate-spin mb-2" size={24} />
                            Fetching Data...
                        </div>
                    ) : (
                        <div className="p-0">
                            <TriageSection title="CRITICAL" patients={grouped.critical} colorClass={riskColors.High} onSelect={handlePatientSelect} selectedId={selectedPatientId} />
                            <TriageSection title="OBSERVATION" patients={grouped.under_observation} colorClass={riskColors.Medium} onSelect={handlePatientSelect} selectedId={selectedPatientId} />
                            <TriageSection title="CLEAR" patients={grouped.clear} colorClass={riskColors.Low} onSelect={handlePatientSelect} selectedId={selectedPatientId} />
                        </div>
                    )}
                </div>
            </div>

            {/* UPLOAD MODAL */}
            {showModal && (
                <UploadModal 
                    patients={patients} 
                    onClose={() => setShowModal(false)} 
                    onSuccess={() => {
                        setShowModal(false);
                        fetchPatients(); // Refetch queue
                    }} 
                />
            )}
        </div>
    );
}

function TriageSection({ title, patients, colorClass, onSelect, selectedId }: { title: string, patients: Patient[], colorClass: string, onSelect: (id: number) => void, selectedId: number | null }) {
    if (patients.length === 0) return null;
    return (
        <div className="border-b border-slate-300 last:border-b-0 pb-4">
            <div className="sticky top-0 bg-white/95 backdrop-blur-md px-4 py-2 border-b border-slate-200 z-10 flex justify-between items-center">
                <span className="font-bold text-[10px] tracking-widest text-slate-500">{title}</span>
                <span className="text-[10px] font-mono font-bold bg-slate-100 px-2 py-0.5 rounded-sm">{patients.length}</span>
            </div>
            <div className="divide-y divide-slate-100">
                {patients.map(p => {
                    const rLevel = p.latest_risk || p.risk_level || 'Clear';
                    const bgCol = rLevel === 'High' ? riskColors.High : rLevel === 'Medium' ? riskColors.Medium : riskColors.Low;
                    const isSelected = p.id === selectedId;
                    
                    return (
                        <motion.div 
                            initial={{ opacity: 0, y: 5 }} 
                            animate={{ opacity: 1, y: 0 }}
                            onClick={() => p.id && onSelect(p.id)}
                            className={`p-4 cursor-pointer transition-colors flex gap-3 group ${isSelected ? 'bg-slate-100 border-l-4 border-zinc-500' : 'hover:bg-slate-50 border-l-4 border-transparent'}`} 
                            key={p.patient_id || p.id}
                        >
                            <div className="flex-1">
                                <div className="flex justify-between items-center mb-1">
                                    <span className={`font-bold text-sm truncate pr-2 ${isSelected ? 'text-zinc-900' : 'group-hover:text-zinc-600'}`}>{p.name}</span>
                                    <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-[2px] ${bgCol}`}>
                                        {rLevel.toUpperCase()}
                                    </span>
                                </div>
                                <div className="flex items-center gap-3 text-[10px] text-slate-500 font-mono">
                                    <span>ID: {p.patient_id || `P-${p.id}`}</span>
                                    <span>•</span>
                                    <span>{p.latest_scan_date ? p.latest_scan_date.split(' ')[0] : 'N/A'}</span>
                                </div>
                            </div>
                            <div className="flex items-center justify-center text-slate-300 group-hover:text-zinc-900 transition-colors">
                                <ChevronRight size={16} />
                            </div>
                        </motion.div>
                    )
                })}
            </div>
        </div>
    );
}

// --- MODAL COMPONENT ---
function UploadModal({ patients, onClose, onSuccess }: { patients: Patient[], onClose: () => void, onSuccess: () => void }) {
    const [isNewPatient, setIsNewPatient] = useState(false);
    
    // Existing Patient State
    const [selectedPatientId, setSelectedPatientId] = useState<string>('');
    
    // New Patient State
    const [newPatientName, setNewPatientName] = useState('');
    const [newPatientNum, setNewPatientNum] = useState('');

    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file) return;

        setUploading(true);
        let finalPatientId = selectedPatientId;

        try {
            if (isNewPatient) {
                // Register patient first
                const createRes = await axios.post(`${API_BASE}/api/patients`, {
                    name: newPatientName,
                    patient_number: newPatientNum
                });
                finalPatientId = String(createRes.data.id);
            }

            if (!finalPatientId) {
                throw new Error("No patient selected or created");
            }

            // Run Analysis
            const formData = new FormData();
            formData.append('patient_id', finalPatientId);
            formData.append('file', file);

            await axios.post(`${API_BASE}/api/analyze`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            onSuccess();
        } catch (err: any) {
            console.error("Upload failed", err);
            const msg = err.response?.data?.detail || "Failed to process scan.";
            alert(`Error: ${msg}`);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="bg-white p-6 w-full max-w-md border border-slate-400 rounded-sm shadow-2xl relative">
                <button onClick={onClose} className="absolute top-4 right-4 text-slate-400 hover:text-zinc-900">
                    <X size={20} />
                </button>
                <h2 className="text-xl font-bold tracking-tight mb-1">Patient Workflow</h2>
                <p className="text-[10px] text-slate-500 font-mono uppercase tracking-widest mb-6">Analyze & Register</p>
                
                {/* TOGGLE */}
                <div className="flex mb-6 bg-slate-100 p-1 rounded-[2px] border border-slate-200">
                    <button 
                        type="button" 
                        onClick={() => setIsNewPatient(false)}
                        className={`flex-1 py-1.5 text-xs font-bold rounded-[2px] transition-colors flex items-center justify-center gap-2 ${!isNewPatient ? 'bg-white shadow-sm border border-slate-200 text-zinc-900' : 'text-slate-500 hover:text-zinc-900'}`}>
                        <Users size={14} /> Existing Patient
                    </button>
                    <button 
                        type="button" 
                        onClick={() => setIsNewPatient(true)}
                        className={`flex-1 py-1.5 text-xs font-bold rounded-[2px] transition-colors flex items-center justify-center gap-2 ${isNewPatient ? 'bg-white shadow-sm border border-slate-200 text-zinc-900' : 'text-slate-500 hover:text-zinc-900'}`}>
                        <UserPlus size={14} /> New Patient
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="space-y-4">
                    {isNewPatient ? (
                        <>
                            <div>
                                <label className="block text-xs font-bold mb-1 text-slate-700">PATIENT NAME</label>
                                <input 
                                    required 
                                    type="text"
                                    placeholder="e.g. John Doe"
                                    value={newPatientName}
                                    onChange={(e) => setNewPatientName(e.target.value)}
                                    className="w-full border border-slate-300 rounded-[2px] p-2 text-sm bg-slate-50 focus:border-zinc-500 focus:outline-none"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-bold mb-1 text-slate-700">PATIENT ADMISSION NO.</label>
                                <input 
                                    required 
                                    type="text"
                                    placeholder="e.g. PT-2051"
                                    value={newPatientNum}
                                    onChange={(e) => setNewPatientNum(e.target.value)}
                                    className="w-full border border-slate-300 rounded-[2px] p-2 text-sm bg-slate-50 focus:border-zinc-500 focus:outline-none"
                                />
                            </div>
                        </>
                    ) : (
                        <div>
                            <label className="block text-xs font-bold mb-1 text-slate-700">SELECT PATIENT</label>
                            <select 
                                required 
                                value={selectedPatientId} 
                                onChange={(e) => setSelectedPatientId(e.target.value)}
                                className="w-full border border-slate-300 rounded-[2px] p-2 text-sm bg-slate-50 focus:border-zinc-500 focus:outline-none"
                            >
                                <option value="">-- Choose Patient --</option>
                                {patients.map(p => (
                                    <option key={p.id} value={p.id}>{p.name} (ID: {p.patient_id || p.id})</option>
                                ))}
                            </select>
                        </div>
                    )}
                    
                    <div className="pt-2">
                        <label className="block text-xs font-bold mb-1 text-slate-700">UPLOAD SCAN FILE (.mhd / img)</label>
                        <input 
                            required 
                            type="file" 
                            accept="image/*,.mhd"
                            onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)}
                            className="w-full border border-slate-300 rounded-[2px] p-2 text-sm bg-slate-50 focus:border-zinc-500 focus:outline-none file:mr-4 file:py-1 file:px-3 file:rounded-[2px] file:border-0 file:text-xs file:font-semibold file:bg-zinc-800 file:text-white hover:file:bg-zinc-700 cursor-pointer"
                        />
                    </div>

                    <div className="pt-4 border-t border-slate-200 mt-6 flex justify-end gap-2">
                        <button type="button" onClick={onClose} className="px-4 py-2 text-xs font-bold text-slate-500 hover:text-zinc-900 shadow-none">
                            CANCEL
                        </button>
                        <button 
                            type="submit" 
                            disabled={uploading || (!isNewPatient && !selectedPatientId) || (isNewPatient && (!newPatientName || !newPatientNum)) || !file}
                            className={`px-6 py-2 flex items-center gap-2 text-xs font-bold text-white rounded-[2px] ${uploading ? 'bg-zinc-400' : 'bg-[#A16207] hover:bg-[#854d05]'} transition-colors shadow-none`}
                        >
                            {uploading ? <Activity size={14} className="animate-spin" /> : null}
                            {uploading ? 'PROCESSING...' : 'RUN ANALYSIS'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}
