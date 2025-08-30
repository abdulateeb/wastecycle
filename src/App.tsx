import { useEffect, useState } from 'react';
import { AnimatePresence } from 'framer-motion';
import Header from './components/Header';
import { Hero } from './components/Hero';
import { HowItWorks } from './components/HowItWorks';
import { ImpactSection } from './components/ImpactSection';
import ImageUpload from './components/ImageUpload';
import Footer from './components/Footer';
import ApiKeyModal from './components/ApiKeyModal';
import { setGeminiApiKey } from './lib/geminiApiKey';

export default function App() {
  const [showChangeKeyModal, setShowChangeKeyModal] = useState(false);

  useEffect(() => {
    document.documentElement.classList.add('dark');
  }, []);

  const handleChangeKeySubmit = (apiKey: string) => {
    setGeminiApiKey(apiKey);
    setShowChangeKeyModal(false);
  };

  return (
    <div className="min-h-screen flex flex-col dark">
      <div className="flex-1 text-white transition-colors">
        <div className="bg-black">
          <Header onChangeKey={() => setShowChangeKeyModal(true)} />
          <Hero />
          <HowItWorks />
          <main>
            <ImageUpload />
          </main>
        </div>
        <ImpactSection />
        <div style={{ background: 'linear-gradient(135deg, rgb(2, 6, 23) 0%, rgb(15, 23, 42) 50%, rgb(30, 41, 59) 100%)' }}>
          <Footer />
        </div>
      </div>

      <AnimatePresence>
        {showChangeKeyModal && (
          <ApiKeyModal
            mode="change"
            onSubmit={handleChangeKeySubmit}
            onCancel={() => setShowChangeKeyModal(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
