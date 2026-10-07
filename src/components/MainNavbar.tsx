import React from 'react';
import type { MainTab } from '../types';
import { Activity, Database, Sparkles, LayoutGrid, Cpu, LineChart, ShieldCheck } from 'lucide-react';

interface MainNavbarProps {
  activeTab: MainTab;
  setActiveTab: (tab: MainTab) => void;
}

export const MainNavbar: React.FC<MainNavbarProps> = ({ activeTab, setActiveTab }) => {
  const mainNavItems = [
    { id: 'home' as MainTab, label: 'HOME', icon: LayoutGrid, tag: 'System Overview' },
    { id: 'component-1' as MainTab, label: 'COMPONENT 1', icon: Cpu, tag: 'Data Pipeline' },
    { id: 'component-2' as MainTab, label: 'COMPONENT 2: Cross-Market & Capital Flow Intelligence Engine', icon: LineChart, tag: 'Market Intelligence' },
    { id: 'component-3' as MainTab, label: 'COMPONENT 3', icon: Activity, tag: 'Risk & Execution' },
    { id: 'component-4' as MainTab, label: 'COMPONENT 4', icon: ShieldCheck, tag: 'Compliance & Audit' },
  ];

  return (
    <header className="bg-[#090b10] border-b border-[#1e2330] sticky top-0 z-40 select-none shadow-xl">
      {/* Top System Title Bar */}
      <div className="max-w-[1700px] mx-auto px-6 py-3 flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-[#181b26]">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-500 to-purple-600 flex items-center justify-center shadow-md shadow-blue-500/20">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-extrabold tracking-tight text-white leading-tight">
                UNIVERSITY RESEARCH MARKET INTELLIGENCE SYSTEM
              </h1>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 text-[10px] font-bold tracking-wider">
                <Sparkles className="w-3 h-3" />
                PROTOTYPE • MOCK DATA
              </span>
            </div>
            <p className="text-[11px] text-gray-400 font-medium">
              4-Component Academic Research Framework Prototype
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-blue-500/10 text-blue-400 font-medium border border-blue-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse"></span>
            Prototype Mode
          </span>
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-500/10 text-amber-400 font-medium border border-amber-500/20">
            <Database className="w-3.5 h-3.5" />
            Mock Data Engine
          </span>
        </div>
      </div>

      {/* Main Navigation Bar (5 TABS ONLY) */}
      <div className="max-w-[1700px] mx-auto px-6">
        <nav className="flex items-center space-x-1 py-2 overflow-x-auto">
          {mainNavItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-2.5 px-5 py-2.5 rounded-lg text-xs font-bold transition-all whitespace-nowrap cursor-pointer ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20 border border-blue-400/30'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-[#121520]'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-gray-400'}`} />
                <span>{item.label}</span>
                <span className={`text-[10px] font-normal px-1.5 py-0.5 rounded ${
                  isActive ? 'bg-blue-700/60 text-blue-100' : 'bg-[#181b26] text-gray-400'
                }`}>
                  {item.tag}
                </span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
