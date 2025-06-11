'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ArrowLeft, BookOpen, FileText, Hash, ChevronLeft, ChevronRight, Menu } from 'lucide-react';
import { HeadingWithContent, HeadingsData } from '@/lib/data-loader';
import { HeadingsTableOfContents } from './headings-table-of-contents';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

interface QuestionViewerProps {
  questionData: HeadingWithContent;
  volumeId: number;
  volumeData?: HeadingsData;
  allQuestions?: HeadingWithContent[];
}

export function QuestionViewer({ questionData, volumeId, volumeData, allQuestions = [] }: QuestionViewerProps) {
  const router = useRouter();
  const [currentIndex, setCurrentIndex] = useState(0);
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

  useEffect(() => {
    // Find current question index
    const index = allQuestions.findIndex(q => q.section_id === questionData.section_id);
    setCurrentIndex(index >= 0 ? index : 0);
  }, [allQuestions, questionData.section_id]);

  const getHeadingIcon = (type: string) => {
    switch (type) {
      case 'heading_major':
        return <BookOpen className="w-6 h-6" />;
      case 'heading_minor':
        return <FileText className="w-5 h-5" />;
      default:
        return <Hash className="w-5 h-5" />;
    }
  };

  const getHeadingStyle = (type: string) => {
    switch (type) {
      case 'heading_major':
        return 'text-2xl md:text-3xl font-bold text-gray-900 leading-tight';
      case 'heading_minor':
        return 'text-xl md:text-2xl font-semibold text-gray-800 leading-tight';
      default:
        return 'text-lg md:text-xl font-medium text-gray-700 leading-tight';
    }
  };

  const getContentTypeStyle = (contentType: string) => {
    switch (contentType) {
      case 'question':
        return 'bg-blue-50 border-l-4 border-blue-400 text-blue-900';
      case 'answer':
        return 'bg-green-50 border-l-4 border-green-400 text-green-900';
      case 'fatwa_ruling':
        return 'border-l-4';
      case 'quran_verse':
        return 'bg-amber-50 border-l-4 border-amber-400 text-amber-900';
      case 'hadith':
        return 'bg-orange-50 border-l-4 border-orange-400 text-orange-900';
      case 'arabic_text':
        return 'bg-gray-50 border-l-4 border-gray-400 text-gray-900 font-arabic text-right';
      default:
        return 'bg-white border-l-4 border-gray-300 text-gray-800';
    }
  };

  const getContentTypeLabel = (contentType: string) => {
    const labels: { [key: string]: string } = {
      'question': 'Question',
      'answer': 'Answer',
      'fatwa_ruling': 'Ruling',
      'quran_verse': 'Qur\'an',
      'hadith': 'Hadith',
      'arabic_text': 'Arabic',
      'paragraph': 'Text',
      'citation': 'Reference',
      'footnote': 'Note'
    };
    return labels[contentType] || contentType.replace('_', ' ');
  };

  const navigateToQuestion = (index: number) => {
    if (index >= 0 && index < allQuestions.length) {
      const question = allQuestions[index];
      router.push(`/volume/${volumeId}/section/${question.section_id}`);
    }
  };

  const handleTOCHeadingClick = (sectionId: number) => {
    // Navigate to the question if it has content
    const heading = volumeData?.headings_to_content_mapping.find(h => h.section_id === sectionId);
    if (heading && heading.content_count > 0) {
      router.push(`/volume/${volumeId}/section/${sectionId}`);
    }
  };

  const previousQuestion = currentIndex > 0 ? allQuestions[currentIndex - 1] : null;
  const nextQuestion = currentIndex < allQuestions.length - 1 ? allQuestions[currentIndex + 1] : null;

  return (
    <div className="w-full min-h-screen overflow-x-hidden">
      {/* Sticky Toggle Button - positioned lower on right side */}
      {volumeData && (
        <div className="fixed top-16 right-2 sm:top-20 sm:right-4 z-[60]">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => setShowTOC(!showTOC)}
            className="shadow-lg bg-white/90 backdrop-blur-sm border-gray-200 text-xs sm:text-sm hover:bg-white/95"
          >
            <Menu className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2" />
            <span className="hidden sm:inline">{showTOC ? 'Hide' : 'Show'} Index</span>
            <span className="sm:hidden">TOC</span>
          </Button>
        </div>
      )}

      {/* Mobile TOC Overlay */}
      {showTOC && volumeData && (
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
                data={volumeData}
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
          {showTOC && volumeData && (
            <div className="hidden md:block w-80 flex-shrink-0">
              <div className="sticky top-24">
                <HeadingsTableOfContents
                  data={volumeData}
                  volumeId={volumeId}
                  onHeadingClick={handleTOCHeadingClick}
                />
              </div>
            </div>
          )}

          {/* Main Content */}
          <div className="flex-1 min-w-0">
            {/* Header with navigation */}
            <div className="mb-4 sm:mb-8 pb-2 sm:pb-4">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pt-2 sm:pt-4 mb-4 sm:mb-6 gap-3 sm:gap-0">
                <Link href={`/volume/${volumeId}`}>
                  <Button variant="outline" size="sm" className="flex items-center gap-2 text-xs sm:text-sm">
                    <ArrowLeft className="w-3 h-3 sm:w-4 sm:h-4" />
                    <span className="hidden sm:inline">Back to Volume {volumeId}</span>
                    <span className="sm:hidden">Vol {volumeId}</span>
                  </Button>
                </Link>
                
                <div className="flex items-center gap-4">
                  {allQuestions.length > 0 && (
                    <div className="text-xs sm:text-sm text-gray-600">
                      Question {currentIndex + 1} of {allQuestions.length}
                    </div>
                  )}
                </div>
              </div>

              {/* Question navigation */}
              {allQuestions.length > 1 && (
                <div className="flex flex-col sm:flex-row items-center justify-between mb-4 sm:mb-6 gap-3 sm:gap-0">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => navigateToQuestion(currentIndex - 1)}
                    disabled={!previousQuestion}
                    className="flex items-center gap-2 text-xs sm:text-sm w-full sm:w-auto"
                  >
                    <ChevronLeft className="w-3 h-3 sm:w-4 sm:h-4" />
                    Previous
                  </Button>
                  
                  <div className="text-center order-first sm:order-none">
                    <div className="text-xs sm:text-sm text-gray-500 mb-1">Navigate Questions</div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => navigateToQuestion(0)}
                        disabled={currentIndex === 0}
                        className="text-xs sm:text-sm"
                      >
                        First
                      </Button>
                      <span className="text-gray-400">|</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => navigateToQuestion(allQuestions.length - 1)}
                        disabled={currentIndex === allQuestions.length - 1}
                        className="text-xs sm:text-sm"
                      >
                        Last
                      </Button>
                    </div>
                  </div>

                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => navigateToQuestion(currentIndex + 1)}
                    disabled={!nextQuestion}
                    className="flex items-center gap-2 text-xs sm:text-sm w-full sm:w-auto"
                  >
                    Next
                    <ChevronRight className="w-3 h-3 sm:w-4 sm:h-4" />
                  </Button>
                </div>
              )}
            </div>

            {/* Main Question Card */}
            <Card className="mb-6 sm:mb-8 shadow-lg border-0 bg-white">
              <CardHeader className="py-3 sm:py-4 bg-gradient-to-r from-blue-50 to-green-50 border-b">
                <div className="flex items-start gap-3 sm:gap-4">
                  <div className="mt-1 text-blue-600 flex-shrink-0">
                    {getHeadingIcon(questionData.heading.type)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <CardTitle className={getHeadingStyle(questionData.heading.type)}>
                      {questionData.heading.text}
                    </CardTitle>
                    {questionData.heading.page_number && (
                      <div className="mt-2 sm:mt-3 text-xs sm:text-sm text-gray-600 bg-white/70 inline-block px-2 sm:px-3 py-1 rounded-full">
                        Page {questionData.heading.page_number}
                      </div>
                    )}
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="p-0">
                <div className="space-y-0">
                  {questionData.content_items.filter(x => x.content_type !== "navigation").map((item, index) => (
                    <div
                      key={index}
                      className={`p-4 sm:p-6 ${getContentTypeStyle(item.content_type)}`}
                    >
                      <div className={`leading-relaxed ${
                        item.content_type === 'arabic_text' ? 'text-base sm:text-lg font-arabic' : 'text-sm sm:text-base'
                      }`}>
                        {item.text}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Related Questions Preview */}
            {(previousQuestion || nextQuestion) && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
                {previousQuestion && (
                  <Link href={`/volume/${volumeId}/section/${previousQuestion.section_id}`}>
                    <Card className="hover:shadow-md transition-shadow cursor-pointer border-gray-200 h-full">
                      <CardHeader className="pb-3">
                        <div className="text-xs sm:text-sm text-gray-500 mb-2">Previous Question</div>
                        <CardTitle className="text-sm sm:text-base font-medium text-gray-800 line-clamp-3">
                          {previousQuestion.heading.text}
                        </CardTitle>
                      </CardHeader>
                    </Card>
                  </Link>
                )}

                {nextQuestion && (
                  <Link href={`/volume/${volumeId}/section/${nextQuestion.section_id}`}>
                    <Card className="hover:shadow-md transition-shadow cursor-pointer border-gray-200 h-full">
                      <CardHeader className="pb-3">
                        <div className="text-xs sm:text-sm text-gray-500 mb-2">Next Question</div>
                        <CardTitle className="text-sm sm:text-base font-medium text-gray-800 line-clamp-3">
                          {nextQuestion.heading.text}
                        </CardTitle>
                      </CardHeader>
                    </Card>
                  </Link>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
