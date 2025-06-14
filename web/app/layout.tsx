import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/next"
import "./globals.css";
import { PostHogProvider } from "./providers";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "Alifta - Islamic Fatwa Collection",
    template: "%s | Alifta - Islamic Fatwa Collection"
  },
  description: "Comprehensive Islamic fatwa collection providing authentic religious guidance and jurisprudence rulings from qualified Islamic scholars. Explore 30 volumes of detailed Islamic teachings and answers to religious questions.",
  keywords: [
    "Islamic fatwa",
    "Islamic jurisprudence",
    "Islamic guidance",
    "religious rulings",
    "Islamic scholars",
    "Quran interpretation",
    "Hadith",
    "Islamic law",
    "Sharia",
    "Islamic questions",
    "fatwa collection",
    "Islamic teachings",
    "religious consultation",
    "Islamic studies",
    "Muslim guidance"
  ],
  authors: [{ name: "Islamic Scholars Committee" }],
  creator: "Islamic Fatwa Committee",
  publisher: "Alifta",
  category: "Religion",
  classification: "Islamic Studies",
  metadataBase: new URL("https://www.al-ifta.com"),
  alternates: {
    canonical: "/",
    languages: {
      "en-US": "/",
      "ar-SA": "/ar",
    },
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    alternateLocale: ["ar_SA"],
    url: "https://www.al-ifta.com",
    siteName: "Alifta - Islamic Fatwa Collection",
    title: "Alifta - Islamic Fatwa Collection",
    description: "Comprehensive Islamic fatwa collection providing authentic religious guidance and jurisprudence rulings from qualified Islamic scholars.",
    images: [
      {
        url: "/og-image.jpg",
        width: 1200,
        height: 630,
        alt: "Alifta - Islamic Fatwa Collection",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Alifta - Islamic Fatwa Collection",
    description: "Comprehensive Islamic fatwa collection providing authentic religious guidance and jurisprudence rulings from qualified Islamic scholars.",
    images: ["/og-image.jpg"],
    creator: "@alifta",
    site: "@alifta",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  verification: {
    google: "your-google-verification-code",
    yandex: "your-yandex-verification-code",
    yahoo: "your-yahoo-verification-code",
  },
  other: {
    "msapplication-TileColor": "#2b5797",
    "theme-color": "#ffffff",
    "apple-mobile-web-app-capable": "yes",
    "apple-mobile-web-app-status-bar-style": "default",
    "format-detection": "telephone=no",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const organizationStructuredData = {
    "@context": "https://schema.org",
    "@type": "Organization",
    "name": "Alifta",
    "description": "Islamic Fatwa Collection providing authentic religious guidance",
    "url": "https://www.al-ifta.com",
    "logo": "https://www.al-ifta.com/logo.png",
    "sameAs": [
      "https://twitter.com/alifta",
      "https://facebook.com/alifta"
    ],
    "contactPoint": {
      "@type": "ContactPoint",
      "contactType": "customer service",
      "availableLanguage": ["English", "Arabic"]
    }
  };

  const websiteStructuredData = {
    "@context": "https://schema.org",
    "@type": "WebSite",
    "name": "Alifta - Islamic Fatwa Collection",
    "description": "Comprehensive Islamic fatwa collection providing authentic religious guidance",
    "url": "https://www.al-ifta.com",
    "inLanguage": ["en", "ar"],
    "about": {
      "@type": "Thing",
      "name": "Islamic Jurisprudence"
    },
    "publisher": {
      "@type": "Organization",
      "name": "Alifta"
    }
  };

  return (
    <html lang="en" dir="ltr">
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify(organizationStructuredData),
          }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify(websiteStructuredData),
          }}
        />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
        <meta httpEquiv="X-UA-Compatible" content="IE=edge" />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {/* <PostHogProvider> */}
          {children}
        {/* </PostHogProvider> */}
        <Analytics />
      </body>
    </html>
  );
}
