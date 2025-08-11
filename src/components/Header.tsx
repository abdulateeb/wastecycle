import { Recycle } from 'lucide-react';

export default function Header() {
  return (
    <header className="w-full py-4 px-6 flex items-center border-b border-gray-700">
      <div className="flex items-center space-x-2">
        <Recycle className="w-8 h-8 text-green-400" />
        <h1 className="text-xl font-bold text-white">AI Material Classification</h1>
      </div>
    </header>
  );
}
