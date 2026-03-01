'use client';

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Line } from '@react-three/drei';
import { ChevronRight, Archive, ClipboardList, Activity, UploadCloud, X, UserPlus, Users } from 'lucide-react';
import { motion } from 'framer-motion';

// --- API FETCHING ---
const API_BASE = 'http://localhost:8000';

type Patient = {
    id?: number;
    patient_id: string; // fallback if api uses string
    name: string;
    risk_level: 'High' | 'Medium' | 'Low' | string;
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

// --- THREE.JS CLINICAL LUNG VISUALIZATION ---
function ClinicalLungModel({ scan }: { scan: ScanRecord | null }) {
    if (!scan) return null;

    // We map the x_coordinate (typically 0-512) to -3 to +3 range for 3D placement
    const noduleX = ((scan.x_coordinate || 256) / 512.0) * 6 - 3;
    // Map mm to relative size
    const noduleR = Math.max(0.15, scan.tumor_diameter_mm / 15.0);

    const isCritical = scan.risk_level === 'High' || scan.risk_level === 'Critical';
    const noduleColor = isCritical ? '#991B1B' : (scan.risk_level === 'Medium' ? '#A16207' : '#166534');

    return (
        <group>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1.2} />
            
            {/* RIGHT LUNG LOBE */}
            <mesh position={[1.4, 0, 0]} scale={[1, 1.8, 1.2]}>
                <sphereGeometry args={[1.5, 32, 32]} />
                <meshStandardMaterial color="#64748B" wireframe opacity={0.2} transparent />
            </mesh>
            
            {/* LEFT LUNG LOBE */}
            <mesh position={[-1.4, 0, 0]} scale={[0.9, 1.7, 1.1]}>
                <sphereGeometry args={[1.5, 32, 32]} />
                <meshStandardMaterial color="#64748B" wireframe opacity={0.2} transparent />
            </mesh>

            {/* TRACHEA / BRONCHI HINT */}
            <Line points={[[0, 3, 0], [0, 1.5, 0]]} color="#64748B" lineWidth={2} opacity={0.5} transparent />
            <Line points={[[0, 1.5, 0], [0.8, 0.5, 0]]} color="#64748B" lineWidth={1.5} opacity={0.5} transparent />
            <Line points={[[0, 1.5, 0], [-0.8, 0.5, 0]]} color="#64748B" lineWidth={1.5} opacity={0.5} transparent />
            
            {/* NODULE */}
            <mesh position={[noduleX, 0, 0.5]}>
                <sphereGeometry args={[noduleR, 16, 16]} />
                <meshStandardMaterial color={noduleColor} emissive={noduleColor} emissiveIntensity={0.6} />
            </mesh>
            
            {/* Tracking Line */}
            <Line
                points={[ [noduleX, 0, 0.5], [noduleX, 2.5, 0.5] ]}
                color={noduleColor}
                lineWidth={1}
            />
            {/* Crosshairs */}
            <Line points={[[0, -3.5, 0.5], [0, 3.5, 0.5]]} color="#64748B" lineWidth={0.5} dashed opacity={0.3} transparent />
            <Line points={[[-4, 0, 0.5], [4, 0, 0.5]]} color="#64748B" lineWidth={0.5} dashed opacity={0.3} transparent />
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
            <div className="flex-1 bg-[#FAFAFA] flex flex-col relative border-r border-slate-400 z-0">
                <div className="absolute top-4 left-4 z-10 font-mono text-xs font-bold text-slate-400">
                    AXIAL VIEW / ISO-SPATIAL RENDER
                </div>
                
                {selectedScan ? (
                    <>
                        <div className="absolute top-4 right-4 z-10 flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${selectedScan.risk_level === 'High' ? 'bg-[#991B1B] animate-pulse' : 'bg-[#166534]'}`}></div>
                            <span className={`font-mono text-[10px] font-bold ${selectedScan.risk_level === 'High' ? 'text-[#991B1B]' : 'text-[#166534]'}`}>
                                NODULE DETECTED
                            </span>
                        </div>
                        <div className="h-full w-full">
                            <Canvas camera={{ position: [0, 0, 8], fov: 45 }}>
                                <color attach="background" args={['#FAFAFA']} />
                                <OrbitControls enableZoom={true} enablePan={true} autoRotate autoRotateSpeed={0.5} />
                                <ClinicalLungModel scan={selectedScan} />
                            </Canvas>
                        </div>
                        <div className="absolute bottom-0 w-full p-4 border-t border-slate-300 bg-white/80 backdrop-blur-sm flex justify-between items-center font-mono text-[10px]">
                            <div>PATIENT: {activePatient?.name}</div>
                            <div>DIAMETER: {selectedScan.tumor_diameter_mm}mm</div>
                            <div>RISK: {selectedScan.risk_level}</div>
                        </div>
                    </>
                ) : (
                    <div className="h-full w-full flex flex-col items-center justify-center text-slate-400 border-2 border-dashed border-slate-300 m-8 w-[calc(100%-4rem)] rounded-[2px]">
                        <UploadCloud size={48} className="mb-4 opacity-50" />
                        <h2 className="text-xl font-bold tracking-tight text-slate-500 mb-2">NO SCAN SELECTED</h2>
                        <p className="text-sm font-mono opacity-70 mb-6">Select a patient from the queue or upload a new scan to get 4D anatomical localizations.</p>
                        <button 
                            onClick={() => setShowModal(true)}
                            className="bg-slate-200 hover:bg-slate-300 text-slate-700 font-bold py-2 px-6 rounded-[2px] text-xs tracking-widest uppercase transition-colors">
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
