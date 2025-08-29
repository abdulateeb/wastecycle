import { Recycle, KeyRound } from 'lucide-react';
import { useStoredApiKey } from '../lib/useStoredApiKey';

interface HeaderProps {
  onChangeKey: () => void;
}

export default function Header({ onChangeKey }: HeaderProps) {
  const hasKey = useStoredApiKey();

  return (
    <header className="w-full py-4 px-6 flex items-center justify-between border-b border-gray-700">
      <div className="flex items-center space-x-2">
        <Recycle className="w-8 h-8 text-green-400" />
        <h1 className="text-xl font-bold text-white">AI Material Classification</h1>
      </div>

      {hasKey && (
        <button
          type="button"
          onClick={onChangeKey}
          className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-green-300 border border-green-500/30 rounded-xl bg-green-500/10 hover:bg-green-500/20 hover:border-green-400/40 transition-colors"
        >
          <KeyRound className="w-4 h-4" />
          Change Key
        </button>
      )}
    </header>
  );
}
