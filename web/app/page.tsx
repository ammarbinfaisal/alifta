import { VolumeIndex } from '@/components/volume-index';
import { getAllVolumes } from '@/lib/data-loader';

export default async function HomePage() {
  const volumes = await getAllVolumes();
  
  return (
    <div className="min-h-screen bg-white">
      <header className="border-b border-gray-200 bg-white">
        <div className="max-w-6xl mx-auto px-4 py-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Majmoo'al-Fatawa of Ibn Bazz
          </h1>
          <p className="text-lg text-gray-600">
            Complete Collection - {volumes.length} Volumes
          </p>
        </div>
      </header>
      <VolumeIndex volumes={volumes} />
    </div>
  );
}