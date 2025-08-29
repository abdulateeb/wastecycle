import { useState, type FormEvent } from 'react';
import { motion } from 'framer-motion';
import { Key, ExternalLink } from 'lucide-react';

type ApiKeyModalMode = 'required' | 'change';

interface ApiKeyModalProps {
  mode?: ApiKeyModalMode;
  onSubmit: (apiKey: string) => void;
  onCancel: () => void;
}

export default function ApiKeyModal({ mode = 'required', onSubmit, onCancel }: ApiKeyModalProps) {
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState<string | null>(null);
  const isChangeMode = mode === 'change';

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = apiKey.trim();
    if (!trimmed) {
      setError('Please enter your Gemini API key');
      return;
    }
    if (trimmed.length < 20) {
      setError('Please enter a valid Gemini API key');
      return;
    }
    onSubmit(trimmed);
  };

  return (
    <motion.div
      className="fixed inset-0 flex items-center justify-center z-50 px-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="fixed inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onCancel}
      />
      <motion.div
        className="relative bg-gradient-to-br from-green-950/80 via-black/90 to-emerald-950/80 backdrop-blur-2xl border border-green-400/20 rounded-2xl shadow-2xl max-w-md w-full p-8"
        initial={{ scale: 0.95, y: 10 }}
        animate={{ scale: 1, y: 0 }}
      >
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-green-500/10 border border-green-500/20">
            <Key className="w-5 h-5 text-green-400" />
          </div>
          <h3 className="text-xl font-bold text-white">
            {isChangeMode ? 'Change Gemini API Key' : 'Gemini API Key Required'}
          </h3>
        </div>

        <p className="text-gray-400 text-sm mb-6">
          {isChangeMode
            ? 'Enter a new Gemini API key to replace the one saved in your browser. The new key will be used for all future analyses.'
            : 'Enter your Google Gemini API key to analyze the uploaded image. Your key is stored locally in your browser and sent directly to Google Gemini for classification.'}
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="gemini-api-key" className="block text-sm font-medium text-gray-300 mb-2">
              Gemini API Key
            </label>
            <input
              id="gemini-api-key"
              type="password"
              value={apiKey}
              onChange={(e) => {
                setApiKey(e.target.value);
                setError(null);
              }}
              placeholder="AIza..."
              autoFocus
              className="w-full px-4 py-3 bg-black/50 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-green-500/50 focus:ring-1 focus:ring-green-500/30 transition-colors"
            />
            {error && (
              <p className="mt-2 text-sm text-red-400">{error}</p>
            )}
          </div>

          <a
            href="https://aistudio.google.com/apikey"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-sm text-green-400 hover:text-green-300 transition-colors"
          >
            Get a free API key from Google AI Studio
            <ExternalLink className="w-3.5 h-3.5" />
          </a>

          <div className="flex gap-3 pt-2">
            <button
              type="button"
              onClick={onCancel}
              className="flex-1 py-3 px-4 border border-white/10 rounded-xl text-gray-300 hover:bg-white/5 transition-colors font-medium"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 py-3 px-4 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 rounded-xl text-white font-medium transition-all shadow-lg shadow-green-500/20"
            >
              {isChangeMode ? 'Save New Key' : 'Start Analysis'}
            </button>
          </div>
        </form>
      </motion.div>
    </motion.div>
  );
}
