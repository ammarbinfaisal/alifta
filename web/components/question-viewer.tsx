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
    <div className="flex gap-6 max-w-7xl mx-auto px-4 py-8">
      {/* Sticky Toggle Button */}
      {volumeData && (
        <div className="fixed top-4 right-4 z-50">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => setShowTOC(!showTOC)}
            className="shadow-lg bg-white"
          >
            <Menu className="w-4 h-4 mr-2" />
            {showTOC ? 'Hide' : 'Show'} Index
          </Button>
        </div>
      )}

      {/* Table of Contents Sidebar */}
      {showTOC && volumeData && (
        <div className="w-80 flex-shrink-0">
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
        <div className="mb-8 pb-4">
          <div className="flex items-center justify-between pt-4 mb-6">
            <Link href={`/volume/${volumeId}`}>
              <Button variant="outline" size="sm" className="flex items-center gap-2">
                <ArrowLeft className="w-4 h-4" />
                Back to Volume {volumeId}
              </Button>
            </Link>
            
            <div className="flex items-center gap-4">
              {allQuestions.length > 0 && (
                <div className="text-sm text-gray-600">
                  Question {currentIndex + 1} of {allQuestions.length}
                </div>
              )}
            </div>
          </div>

          {/* Question navigation */}
          {allQuestions.length > 1 && (
            <div className="flex items-center justify-between mb-6">
              <Button
                variant="outline"
                size="sm"
                onClick={() => navigateToQuestion(currentIndex - 1)}
                disabled={!previousQuestion}
                className="flex items-center gap-2"
              >
                <ChevronLeft className="w-4 h-4" />
                Previous
              </Button>
              
              <div className="text-center">
                <div className="text-sm text-gray-500 mb-1">Navigate Questions</div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => navigateToQuestion(0)}
                    disabled={currentIndex === 0}
                  >
                    First
                  </Button>
                  <span className="text-gray-400">|</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => navigateToQuestion(allQuestions.length - 1)}
                    disabled={currentIndex === allQuestions.length - 1}
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
                className="flex items-center gap-2"
              >
                Next
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
          )}
        </div>

        {/* Main Question Card */}
        <Card className="mb-8 shadow-lg border-0 bg-white">
          <CardHeader className="py-4 bg-gradient-to-r from-blue-50 to-green-50 border-b">
            <div className="flex items-start gap-4">
              <div className="mt-1 text-blue-600">
                {getHeadingIcon(questionData.heading.type)}
              </div>
              <div className="flex-1">
                <CardTitle className={getHeadingStyle(questionData.heading.type)}>
                  {questionData.heading.text}
                </CardTitle>
                {questionData.heading.page_number && (
                  <div className="mt-3 text-sm text-gray-600 bg-white/70 inline-block px-3 py-1 rounded-full">
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
                  className={`p-6 ${getContentTypeStyle(item.content_type)}`}
                >
                  <div className={`leading-relaxed ${
                    item.content_type === 'arabic_text' ? 'text-lg font-arabic' : 'text-base'
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
          <div className="grid md:grid-cols-2 gap-6">
            {previousQuestion && (
              <Link href={`/volume/${volumeId}/section/${previousQuestion.section_id}`}>
                <Card className="hover:shadow-md transition-shadow cursor-pointer border-gray-200 h-full">
                  <CardHeader className="pb-3">
                    <div className="text-sm text-gray-500 mb-2">Previous Question</div>
                    <CardTitle className="text-base font-medium text-gray-800 line-clamp-3">
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
                    <div className="text-sm text-gray-500 mb-2">Next Question</div>
                    <CardTitle className="text-base font-medium text-gray-800 line-clamp-3">
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
  );
}
