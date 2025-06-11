import { notFound } from 'next/navigation';
import { getHeadingWithContent, getAllHeadingsWithContent, getVolumeData } from '@/lib/data-loader';
import { QuestionViewer } from '@/components/question-viewer';
import { VolumeNavigation } from '@/components/volume-navigation';

interface QuestionPageProps {
  params: { id: string; sectionId: string };
}

// Generate static params for all questions in all volumes
export async function generateStaticParams() {
  const params = [];
  
  // For now, generate for volume 1 (can be extended for all volumes)
  for (let volumeId = 1; volumeId <= 1; volumeId++) {
    try {
      const headings = await getAllHeadingsWithContent(volumeId);
      for (const heading of headings) {
        if (heading.content_count > 0) {
          params.push({
            id: volumeId.toString(),
            sectionId: heading.section_id.toString()
          });
        }
      }
    } catch (error) {
      // Skip volumes that don't exist
      continue;
    }
  }
  
  return params;
}

export default async function QuestionPage({ params }: QuestionPageProps) {
  const volumeId = parseInt(params.id);
  const sectionId = parseInt(params.sectionId);
  
  if (isNaN(volumeId) || isNaN(sectionId) || volumeId < 1 || volumeId > 30) {
    notFound();
  }

  const questionData = await getHeadingWithContent(volumeId, sectionId);
  const volumeData = await getVolumeData(volumeId);
  
  if (!questionData || questionData.content_count === 0 || !volumeData) {
    notFound();
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <VolumeNavigation currentVolumeId={volumeId} />
      <QuestionViewer
        questionData={questionData}
        volumeId={volumeId}
        volumeData={volumeData}
      />
    </div>
  );
}
