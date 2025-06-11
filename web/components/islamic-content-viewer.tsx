'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ChevronRight, BookOpen, FileText, Hash, Menu } from 'lucide-react';
import { HeadingsData, HeadingWithContent, getAllHeadingsWithContent } from '@/lib/data-loader';
import { HeadingsTableOfContents } from './headings-table-of-contents';
import Link from 'next/link';

interface IslamicContentViewerProps {
  data: HeadingsData;
  volumeId: number;
}

interface HeadingItemProps {
  heading: HeadingWithContent;
  volumeId: number;
}

function HeadingItem({ heading, volumeId }: HeadingItemProps) {
  const getHeadingIcon = (type: string) => {
    switch (type) {
      case 'heading_major':
        return <BookOpen className="w-4 h-4 sm:w-5 sm:h-5" />;
      case 'heading_minor':
        return <FileText className="w-3 h-3 sm:w-4 sm:h-4" />;
      default:
        return <Hash className="w-3 h-3 sm:w-4 sm:h-4" />;
    }
  };

  const getHeadingStyle = (type: string) => {
    switch (type) {
      case 'heading_major':
        return 'text-lg sm:text-xl font-bold text-gray-900';
      case 'heading_minor':
        return 'text-base sm:text-lg font-semibold text-gray-800';
      default:
        return 'text-sm sm:text-base font-medium text-gray-700';
    }
  };

  const getCardStyle = (type: string) => {
    switch (type) {
      case 'heading_major':
        return 'border-l-4 border-l-blue-500 bg-blue-50/50 hover:bg-blue-100/50';
      case 'heading_minor':
        return 'border-l-4 border-l-green-500 bg-green-50/50 hover:bg-green-100/50';
      default:
        return 'border-l-4 border-l-gray-400 bg-gray-50/50 hover:bg-gray-100/50';
    }
  };

  const CardContent = heading.content_count > 0 ? (
    <Link href={`/volume/${volumeId}/section/${heading.section_id}`}>
      <Card className={`mb-3 sm:mb-4 transition-all duration-200 cursor-pointer ${getCardStyle(heading.heading.type)}`}>
        <CardHeader className="pb-2 sm:pb-3 p-3 sm:p-6">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-2 sm:gap-3 flex-1 min-w-0">
              <div className="mt-1 flex-shrink-0">
                {getHeadingIcon(heading.heading.type)}
              </div>
              <div className="flex-1 min-w-0">
                <CardTitle className={`${getHeadingStyle(heading.heading.type)} line-clamp-3`}>
                  {heading.heading.text}
                </CardTitle>
                <div className="flex items-center gap-2 mt-1 sm:mt-2">
                  {heading.heading.page_number && (
                    <Badge variant="outline" className="text-xs">
                      Page {heading.heading.page_number}
                    </Badge>
                  )}
                </div>
                <div className="mt-1 sm:mt-2 text-xs sm:text-sm text-gray-600">
                  Click to view full content →
                </div>
              </div>
            </div>
            <ChevronRight className="w-4 h-4 sm:w-5 sm:h-5 text-gray-400 mt-1 flex-shrink-0" />
          </div>
        </CardHeader>
      </Card>
    </Link>
  ) : (
    <Card className={`mb-3 sm:mb-4 opacity-60 ${getCardStyle(heading.heading.type)}`}>
      <CardHeader className="pb-2 sm:pb-3 p-3 sm:p-6">
        <div className="flex items-start gap-2 sm:gap-3">
          <div className="mt-1 flex-shrink-0">
            {getHeadingIcon(heading.heading.type)}
          </div>
          <div className="flex-1 min-w-0">
            <CardTitle className={`${getHeadingStyle(heading.heading.type)} line-clamp-3`}>
              {heading.heading.text}
            </CardTitle>
            <div className="flex items-center gap-2 mt-1 sm:mt-2">
              <Badge variant="outline" className="text-xs">
                No content
              </Badge>
              {heading.heading.page_number && (
                <Badge variant="outline" className="text-xs">
                  Page {heading.heading.page_number}
                </Badge>
              )}
            </div>
          </div>
        </div>
      </CardHeader>
    </Card>
  );

  return CardContent;
}

export function IslamicContentViewer({ data, volumeId }: IslamicContentViewerProps) {
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set());
  const [filterType, setFilterType] = useState<string>('all');
  const [showTOC, setShowTOC] = useState<boolean>(false);

  // Set initial TOC visibility based on screen size
  useEffect(() => {
    const checkScreenSize = () => {
      const isMobile = window.innerWidth < 768; // md breakpoint
      setShowTOC(!isMobile);
    };

    // Check on mount
    checkScreenSize();

    // Listen for resize events
    window.addEventListener('resize', checkScreenSize);
    return () => window.removeEventListener('resize', checkScreenSize);
  }, []);

  const toggleSection = (sectionId: number) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(sectionId)) {
      newExpanded.delete(sectionId);
    } else {
      newExpanded.add(sectionId);
    }
    setExpandedSections(newExpanded);
  };

  const expandAll = () => {
    const allSectionIds = data.headings_to_content_mapping
      .filter(h => h.content_count > 0)
      .map(h => h.section_id);
    setExpandedSections(new Set(allSectionIds));
  };

  const collapseAll = () => {
    setExpandedSections(new Set());
  };

  const handleTOCHeadingClick = (sectionId: number) => {
    // Scroll to heading and expand it if it has content
    const element = document.getElementById(`heading-${sectionId}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      
      // Auto-expand if it has content
      const heading = data.headings_to_content_mapping.find(h => h.section_id === sectionId);
      if (heading && heading.content_count > 0) {
        const newExpanded = new Set(expandedSections);
        newExpanded.add(sectionId);
        setExpandedSections(newExpanded);
      }
    }
  };

  const filteredHeadings = data.headings_to_content_mapping.filter(heading => {
    if (filterType === 'all') return true;
    return heading.heading.type === filterType;
  });

  const headingTypes = data.by_type ? Object.keys(data.by_type) : [];

  return (
    <div className="w-full min-h-screen overflow-x-hidden">
      {/* Sticky Toggle Button - positioned lower on right side */}
      <div className="fixed top-16 right-2 sm:top-20 sm:right-4 z-[60]">
        <Button 
          variant="outline" 
          size="sm" 
          onClick={() => setShowTOC(!showTOC)}
          className="shadow-lg bg-white/90 backdrop-blur-sm border-gray-200 text-xs sm:text-sm hover:bg-white/95"
        >
          <Menu className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2" />
          <span className="hidden sm:inline">{showTOC ? 'Hide' : 'Show'} TOC</span>
          <span className="sm:hidden">TOC</span>
        </Button>
      </div>

      {/* Mobile TOC Overlay */}
      {showTOC && (
        <>
          {/* Backdrop with blur */}
          <div 
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[45] md:hidden"
            onClick={() => setShowTOC(false)}
          />
          
          {/* Mobile TOC */}
          <div className="fixed inset-y-0 left-0 w-4/5 max-w-sm bg-white/95 backdrop-blur-md z-[50] md:hidden overflow-y-auto shadow-2xl">
            <div className="p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900">Table of Contents</h3>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={() => setShowTOC(false)}
                  className="hover:bg-gray-100"
                >
                  ×
                </Button>
              </div>
              <HeadingsTableOfContents
                data={data}
                volumeId={volumeId}
                onHeadingClick={(sectionId) => {
                  handleTOCHeadingClick(sectionId);
                  setShowTOC(false); // Close TOC on mobile after selection
                }}
              />
            </div>
          </div>
        </>
      )}

      <div className="max-w-7xl mx-auto px-2 sm:px-4 py-4 sm:py-8">
        <div className="flex gap-6">
          {/* Desktop TOC Sidebar */}
          {showTOC && (
            <div className="hidden md:block w-80 flex-shrink-0">
              <div className="sticky top-24">
                <HeadingsTableOfContents
                  data={data}
                  volumeId={volumeId}
                  onHeadingClick={handleTOCHeadingClick}
                />
              </div>
            </div>
          )}

          {/* Main Content */}
          <div className="flex-1 min-w-0">
            {/* Header */}
            <div className="mb-6 sm:mb-8 pb-2 sm:pb-4">
              <div className="flex items-center justify-between pt-2 sm:pt-4">
                <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-gray-900">
                  Volume {volumeId} - Headings & Content
                </h1>
              </div>
            </div>

            {/* Headings list */}
            <div className="space-y-3 sm:space-y-4">
              {filteredHeadings.map((heading) => (
                <div key={heading.section_id} id={`heading-${heading.section_id}`}>
                  <HeadingItem
                    heading={heading}
                    volumeId={volumeId}
                  />
                </div>
              ))}
            </div>

            {filteredHeadings.length === 0 && (
              <div className="text-center py-8 sm:py-12">
                <div className="text-gray-500 text-base sm:text-lg">
                  No sections found for the selected filter.
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
