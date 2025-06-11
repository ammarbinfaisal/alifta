import { notFound } from 'next/navigation';
import { Metadata } from 'next';
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

// Generate metadata for SEO
export async function generateMetadata({ params }: VolumePageProps): Promise<Metadata> {
  const volumeId = parseInt((await params).id);
  
  if (isNaN(volumeId) || volumeId < 1 || volumeId > 30) {
    return {
      title: 'Volume Not Found',
      description: 'The requested volume could not be found.',
    };
  }

  const volumeData = await getVolumeData(volumeId);
  
  if (!volumeData) {
    return {
      title: 'Volume Not Found',
      description: 'The requested volume could not be found.',
    };
  }

  const title = `Volume ${volumeId} - Islamic Fatwa Collection`;
  const description = `Explore Volume ${volumeId} of the comprehensive Islamic fatwa collection. Contains ${volumeData.total_headings} sectionsand ${volumeData.total_sections} sections covering various Islamic jurisprudence topics and religious guidance.`;
  const url = `https://al-ifta.com/volume/${volumeId}`;

  return {
    title,
    description,
    keywords: [
      'Islamic fatwa',
      'Islamic jurisprudence',
      'Islamic guidance',
      'religious rulings',
      'Islamic scholars',
      'Quran interpretation',
      'Hadith',
      'Islamic law',
      'Sharia',
      'Islamic questions',
      `Volume ${volumeId}`,
      'fatwa collection',
      'Islamic teachings',
      'religious consultation'
    ],
    authors: [{ name: 'Islamic Scholars' }],
    creator: 'Islamic Fatwa Committee',
    publisher: 'Alifta',
    category: 'Religion',
    classification: 'Islamic Studies',
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
    openGraph: {
      type: 'website',
      locale: 'en_US',
      alternateLocale: ['ar_SA'],
      url,
      title,
      description,
      siteName: 'Alifta - Islamic Fatwa Collection',
      images: [
        {
          url: '/og-image.jpg',
          width: 1200,
          height: 630,
          alt: `Volume ${volumeId} - Islamic Fatwa Collection`,
        },
      ],
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
      images: ['/og-image.jpg'],
      creator: '@alifta',
      site: '@alifta',
    },
    alternates: {
      canonical: url,
      languages: {
        'en-US': url,
      },
    },
    other: {
      'article:section': 'Islamic Studies',
      'article:tag': 'Islamic Fatwa, Religious Guidance, Islamic Law',
      'dc.title': title,
      'dc.description': description,
      'dc.type': 'Text',
      'dc.format': 'text/html',
      'dc.language': 'en',
      'dc.subject': 'Islamic Jurisprudence',
      'dc.coverage': 'Worldwide',
      'dc.rights': 'All rights reserved',
    },
  };
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

  // Structured data for search engines
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "Book",
    "name": `Volume ${volumeId} - Islamic Fatwa Collection`,
    "description": `Comprehensive Islamic fatwa collection Volume ${volumeId} containing ${volumeData.total_sections} sections of religious guidance and jurisprudence.`,
    "author": {
      "@type": "Organization",
      "name": "Islamic Scholars Committee"
    },
    "publisher": {
      "@type": "Organization",
      "name": "Alifta",
      "url": "https://al-ifta.com"
    },
    "genre": "Religious Text",
    "inLanguage": ["en", "ar"],
    "about": [
      {
        "@type": "Thing",
        "name": "Islamic Jurisprudence"
      },
      {
        "@type": "Thing", 
        "name": "Islamic Law"
      },
      {
        "@type": "Thing",
        "name": "Religious Guidance"
      }
    ],
    "numberOfPages": volumeData.total_sections,
    "bookFormat": "EBook",
    "url": `https://al-ifta.com/volume/${volumeId}`,
    "mainEntity": {
      "@type": "WebPage",
      "name": `Volume ${volumeId}`,
      "description": `Islamic fatwa collection volume containing religious rulings and guidance`,
      "url": `https://al-ifta.com/volume/${volumeId}`
    }
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(structuredData),
        }}
      />
      <div className="min-h-screen bg-white">
        <VolumeNavigation currentVolumeId={volumeId} />
        <IslamicContentViewer data={volumeData} volumeId={volumeId} />
      </div>
    </>
  );
}
