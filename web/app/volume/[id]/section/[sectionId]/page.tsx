import { notFound } from 'next/navigation';
import { Metadata } from 'next';
import { getHeadingWithContent, getAllHeadingsWithContent, getVolumeData } from '@/lib/data-loader';
import { QuestionViewer } from '@/components/question-viewer';
import { VolumeNavigation } from '@/components/volume-navigation';

interface QuestionPageProps {
  params: Promise<{ id: string; sectionId: string }>;
}


export async function generateMetadata({ params }: QuestionPageProps): Promise<Metadata> {
  const p = await params;
  const volumeId = parseInt(p.id);
  const sectionId = parseInt(p.sectionId);

  if (isNaN(volumeId) || isNaN(sectionId) || volumeId < 1 || volumeId > 30) {
    return {
      title: 'Question Not Found - Al-Ifta',
      description: 'The requested Islamic question could not be found.'
    };
  }

  try {
    const questionData = await getHeadingWithContent(volumeId, sectionId);
    const volumeData = await getVolumeData(volumeId);

    if (!questionData || questionData.content_count === 0 || !volumeData) {
      return {
        title: 'Question Not Found - Al-Ifta',
        description: 'The requested Islamic question could not be found.'
      };
    }

    // Create SEO-friendly title and description
    const questionTitle = questionData.heading.text || `Question ${sectionId}`;

    // Truncate heading for meta description if too long
    const description = questionData.heading.text
      ? `${questionData.heading.text.substring(0, 150)}${questionData.heading.text.length > 150 ? '...' : ''} - Islamic Q&A from Volume ${volumeId}`
      : `Islamic Q&A from Volume ${volumeId}, Section ${sectionId}`;
    const title = `${questionTitle} - Volume ${volumeId} | Al-Ifta Islamic Q&A`;

    return {
      title,
      description,
      keywords: [
        'Islamic Q&A',
        'Islamic questions',
        'Islamic answers',
        'Islamic jurisprudence',
        'Fatwa',
        'Islamic guidance',
        +       `Volume ${volumeId}`,
        'Islamic scholarship',
        'ibn baz',
        'bin baz',
      ].join(', '),
      openGraph: {
        title,
        description,
        type: 'article',
        url: `/volume/${volumeId}/section/${sectionId}`,
        siteName: 'Al-Ifta - Majmu al Fatawa of Ibn Baz'
      },
      twitter: {
        card: 'summary',
        title,
        description
      },
      robots: {
        index: true,
        follow: true,
        googleBot: {
          index: true,
          follow: true,
          'max-video-preview': -1,
          'max-image-preview': 'large',
          'max-snippet': -1,
        },
      },
      alternates: {
        canonical: `/volume/${volumeId}/section/${sectionId}`
      }
    };
  } catch (error) {
    return {
      title: 'Islamic Q&A - Al-Ifta',
      description: 'Islamic questions and answers collection'
    };
  }
}

// Generate static params for all questions in all volumes
export async function generateStaticParams() {
  const params = [];

  // For now, generate for volume 1 (can be extended for all volumes)
  for (let volumeId = 1; volumeId <= 5; volumeId++) {
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
  const p = await params;
  const volumeId = parseInt(p.id);
  const sectionId = parseInt(p.sectionId);

  if (isNaN(volumeId) || isNaN(sectionId) || volumeId < 1 || volumeId > 30) {
    notFound();
  }

  const questionData = await getHeadingWithContent(volumeId, sectionId);
  const volumeData = await getVolumeData(volumeId);

  if (!questionData || questionData.content_count === 0 || !volumeData) {
    notFound();
  }

  const structuredData = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "BreadcrumbList",
        "itemListElement": [
          {
            "@type": "ListItem",
            "position": 1,
            "name": "Home",
            "item": "https://www.al-ifta.com"
          },
          {
            "@type": "ListItem",
            "position": 2,
            "name": `Volume ${volumeId}`,
            "item": `https://www.al-ifta.com/volume/${volumeId}`
          },
          {
            "@type": "ListItem",
            "position": 3,
            "name": questionData.heading.text || `Question ${sectionId}`,
            "item": `https://www.al-ifta.com/volume/${volumeId}/section/${sectionId}`
          }
        ]
      },
      {
        "@type": "WebPage",
        "name": `${questionData.heading.text || `Question ${sectionId}`} | Volume ${volumeId} | Islamic Q&A`,
        "description": questionData.heading.text
          ? `${questionData.heading.text.substring(0, 150)}${questionData.heading.text.length > 150 ? '...' : ''} - Islamic Q&A from Volume ${volumeId}`
          : `Islamic Q&A from Volume ${volumeId}, Section ${sectionId}`,
        "url": `https://www.al-ifta.com/volume/${volumeId}/section/${sectionId}`,
        "mainEntity": {
          "@type": "Question",
          "name": questionData.heading.text || `Question ${sectionId}`,
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "Islamic scholarly response from Majmu' al Fatawa of Imam ibn Baz"
          }
        }
      }
    ]
  };

  // Get all questions with content for navigation
  const allQuestions = await getAllHeadingsWithContent(volumeId);
  const questionsWithContent = allQuestions.filter(h => h.content_count > 0);

  return (
    <div className="min-h-screen bg-gray-50">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(structuredData),
        }}
      />
      <VolumeNavigation currentVolumeId={volumeId} />
      <QuestionViewer
        questionData={questionData}
        volumeId={volumeId}
        volumeData={volumeData}
        allQuestions={questionsWithContent}
      />
    </div>
  );
}
