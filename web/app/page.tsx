import { Metadata } from 'next';
import { VolumeIndex } from '@/components/volume-index';
import { getAllVolumes } from '@/lib/data-loader';

// Generate metadata for the home page
export async function generateMetadata(): Promise<Metadata> {
  const volumes = await getAllVolumes();
  const totalVolumes = volumes.length;
  
  const title = "Majmoo'al-Fatawa of Ibn Bazz - Complete Islamic Fatwa Collection";
  const description = `Explore the complete collection of Majmoo'al-Fatawa by Sheikh Abdul Aziz Ibn Bazz. Access ${totalVolumes} volumes of authentic Islamic fatwas, religious rulings, and scholarly guidance on Islamic jurisprudence, worship, and daily life matters.`;
  const url = "https://www.al-ifta.com";

  return {
    title,
    description,
    keywords: [
      "Ibn Bazz",
      "Abdul Aziz Ibn Bazz",
      "Majmoo al-Fatawa",
      "Islamic fatwa",
      "Islamic jurisprudence",
      "Islamic guidance",
      "religious rulings",
      "Islamic scholars",
      "Saudi scholars",
      "Quran interpretation",
      "Hadith",
      "Islamic law",
      "Sharia",
      "Islamic questions",
      "fatwa collection",
      "Islamic teachings",
      "religious consultation",
      "Islamic studies",
      "Muslim guidance",
      "Grand Mufti",
      "Islamic authority"
    ],
    authors: [{ name: "Sheikh Abdul Aziz Ibn Bazz" }],
    creator: "Sheikh Abdul Aziz Ibn Bazz",
    publisher: "Alifta",
    category: "Religion",
    classification: "Islamic Studies",
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
          url: '/og-home-image.jpg',
          width: 1200,
          height: 630,
          alt: "Majmoo'al-Fatawa of Ibn Bazz - Complete Collection",
        },
      ],
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
      images: ['/og-home-image.jpg'],
      creator: '@alifta',
      site: '@alifta',
    },
    alternates: {
      canonical: url,
      languages: {
        'en-US': url,
        'ar-SA': `${url}/ar`,
      },
    },
    other: {
      'article:section': 'Islamic Studies',
      'article:tag': 'Ibn Bazz, Islamic Fatwa, Religious Guidance, Islamic Law',
      'dc.title': title,
      'dc.description': description,
      'dc.type': 'Collection',
      'dc.format': 'text/html',
      'dc.language': 'en',
      'dc.subject': 'Islamic Jurisprudence',
      'dc.coverage': 'Worldwide',
      'dc.rights': 'All rights reserved',
      'dc.creator': 'Sheikh Abdul Aziz Ibn Bazz',
    },
  };
}

export default async function HomePage() {
  const volumes = await getAllVolumes();
  
  // Structured data for the home page
  const collectionStructuredData = {
    "@context": "https://schema.org",
    "@type": "Collection",
    "name": "Majmoo'al-Fatawa of Ibn Bazz",
    "description": `Complete collection of Islamic fatwas by Sheikh Abdul Aziz Ibn Bazz containing ${volumes.length} volumes of religious guidance and jurisprudence.`,
    "author": {
      "@type": "Person",
      "name": "Sheikh Abdul Aziz Ibn Bazz",
      "jobTitle": "Grand Mufti of Saudi Arabia",
      "description": "Renowned Islamic scholar and former Grand Mufti of Saudi Arabia"
    },
    "publisher": {
      "@type": "Organization",
      "name": "Alifta",
      "url": "https://www.al-ifta.com"
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
      },
      {
        "@type": "Thing",
        "name": "Islamic Fatwas"
      }
    ],
    "numberOfItems": volumes.length,
    "url": "https://www.al-ifta.com",
    "mainEntity": {
      "@type": "WebPage",
      "name": "Majmoo'al-Fatawa Collection",
      "description": "Complete collection of Islamic fatwas and religious rulings",
      "url": "https://www.al-ifta.com"
    },
    "hasPart": volumes.map(volume => ({
      "@type": "Book",
      "name": `Volume ${volume.id}`,
      "url": `https://www.al-ifta.com/volume/${volume.id}`,
      "author": {
        "@type": "Person",
        "name": "Sheikh Abdul Aziz Ibn Bazz"
      }
    }))
  };

  const breadcrumbStructuredData = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    "itemListElement": [
      {
        "@type": "ListItem",
        "position": 1,
        "name": "Home",
        "item": "https://www.al-ifta.com"
      }
    ]
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(collectionStructuredData),
        }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(breadcrumbStructuredData),
        }}
      />
      <div className="min-h-screen bg-white">
        <header className="border-b border-gray-200 bg-white">
          <div className="max-w-6xl mx-auto px-4 py-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-2">
              Majmoo'al-Fatawa of Ibn Bazz
            </h1>
            <p className="text-lg text-gray-600">
              Collection - {volumes.length} / 30 Volumes
            </p>
            <a className='underline' href='https://github.com/ammarbinfaisal/alifta'>
              Source Code
            </a>
          </div>
        </header>
        <VolumeIndex volumes={volumes} />
      </div>
    </>
  );
}
