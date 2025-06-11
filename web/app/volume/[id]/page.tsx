import { notFound } from 'next/navigation';
import { getVolumeData, getAllVolumeIds } from '@/lib/data-loader';
import { IslamicContentViewer } from '@/components/islamic-content-viewer';
import { VolumeNavigation } from '@/components/volume-navigation';

interface VolumePageProps {
  params: Promise<{ id: string }>;
}

// Generate static params for all 30 volumes
export async function generateStaticParams() {
  const volumeIds = await getAllVolumeIds();
  return volumeIds.map((id) => ({ id: id.toString() }));
}

export default async function VolumePage({ params }: VolumePageProps) {
  const volumeId = parseInt((await params).id);
  
  if (isNaN(volumeId) || volumeId < 1 || volumeId > 30) {
    notFound();
  }

  const volumeData = await getVolumeData(volumeId);
  
  if (!volumeData) {
    notFound();
  }

  return (
    <div className="min-h-screen bg-white">
      <VolumeNavigation currentVolumeId={volumeId} />
      <IslamicContentViewer data={volumeData} volumeId={volumeId} />
    </div>
  );
}