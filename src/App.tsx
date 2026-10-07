import { useState } from 'react';
import type { MainTab } from './types';
import { MainNavbar } from './components/MainNavbar';
import { HomePage } from './pages/HomePage';
import { Component1Page } from './pages/Component1Page';
import { Component2Page } from './pages/Component2Page';
import { Component3Page } from './pages/Component3Page';
import { Component4Page } from './pages/Component4Page';
import { ShieldAlert } from 'lucide-react';

export function App() {
  const [activeTab, setActiveTab] = useState<MainTab>('home');

  const renderActiveMainTab = () => {
    switch (activeTab) {
      case 'home':
        return <HomePage onNavigateToTab={(tab) => setActiveTab(tab)} />;
      case 'component-1':
        return <Component1Page />;
      case 'component-2':
        return <Component2Page />;
      case 'component-3':
        return <Component3Page />;
      case 'component-4':
        return <Component4Page />;
      default:
        return <HomePage onNavigateToTab={(tab) => setActiveTab(tab)} />;
    }
  };

  return (
    <div className="min-h-screen bg-[#0b0c10] text-[#e2e8f0] font-sans antialiased selection:bg-blue-500 selection:text-white flex flex-col justify-between">
      <div>
        {/* Top 5-Tab Main Navigation Bar */}
        <MainNavbar activeTab={activeTab} setActiveTab={setActiveTab} />

        {/* Main Content Area */}
        <main className="max-w-[1700px] mx-auto p-6 w-full">
          {renderActiveMainTab()}
        </main>
      </div>

      {/* Global Academic Disclaimer Footer */}
      <footer className="border-t border-[#1e2330] bg-[#08090d] p-4 text-center text-xs text-gray-400 space-y-1 mt-12">
        <div className="flex items-center justify-center gap-1.5 text-amber-400 font-medium">
          <ShieldAlert className="w-3.5 h-3.5" />
          <span>Academic Research Disclaimer</span>
        </div>
        <p className="max-w-4xl mx-auto text-[11px] text-gray-400">
          Frontend prototype for academic research. All displayed values and model outputs are simulated and are not financial advice.
        </p>
      </footer>
    </div>
  );
}

export default App;
