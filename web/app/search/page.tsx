'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Search, BookOpen, FileText, Hash, ChevronLeft, ChevronRight, Loader2, Filter, X } from 'lucide-react';
import Link from 'next/link';

interface ContentItem {
  content_type: string;
  matched_tokens: string[];
  snippet: string;
  value: string;
}

interface SearchHit {
  document: {
    content_count: number;
    content_items: ContentItem[];
    section_id?: number;
    vol?: number;
    heading?: {
      text: string;
      type: string;
      page_number?: number;
    };
  };
  highlight: {
    [key: string]: any;
  };
  highlights: string[];
  text_match: number;
  text_match_info?: {
    best_field_score: string;
    best_field_weight: number;
    fields_matched: number;
    num_tokens_dropped: number;
    score: number;
    tokens_matched: number;
    typo_prefix_score: number;
  };
}

interface QueryInfo {
  original_query: string;
  processed_query: string;
  spelling_corrected: boolean;
  corrected_word?: string;
}

interface SearchResponse {
  facet_counts: any[];
  found: number;
  page: number;
  query_info: QueryInfo;
  results: SearchHit[];
  text_match: number;
  search_time_ms?: number;
  error?: string;
}

export default function SearchPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  
  // Combined search state
  const [searchState, setSearchState] = useState({
    query: '',
    debouncedQuery: '',
    results: [] as SearchHit[],
    loading: false,
    error: null as string | null
  });

  // Combined pagination state
  const [paginationState, setPaginationState] = useState({
    currentPage: 1,
    totalResults: 0,
    searchTime: 0
  });

  // Combined UI state
  const [uiState, setUiState] = useState({
    showFilters: false,
    isInitialized: false
  });

  // Filter state
  const [selectedVolume, setSelectedVolume] = useState<string>('');
  
  const perPage = 10;
  const totalPages = Math.ceil(paginationState.totalResults / perPage);

  // Initialize state from URL parameters
  useEffect(() => {
    const urlQuery = searchParams.get('q') || '';
    const urlPage = parseInt(searchParams.get('page') || '1');
    const urlVolume = searchParams.get('volume') || '';
    
    setSearchState(prev => ({
      ...prev,
      query: urlQuery,
      debouncedQuery: urlQuery
    }));
    setPaginationState(prev => ({
      ...prev,
      currentPage: urlPage
    }));
    setSelectedVolume(urlVolume);
    setUiState(prev => ({
      ...prev,
      isInitialized: true
    }));
    
    // Perform search if there's a query in the URL
    if (urlQuery.trim()) {
      performSearch(urlQuery, urlPage, urlVolume);
    }
  }, []);

  // Function to update URL parameters
  const updateURL = useCallback((searchQuery: string, page: number, volume: string) => {
    const params = new URLSearchParams();

    console.log({page})
    
    if (searchQuery.trim()) {
      params.set('q', searchQuery.trim());
    }
    
    if (page > 1) {
      params.set('page', page.toString());
    }
    
    if (volume) {
      params.set('volume', volume);
    }
    
    const newURL = params.toString() ? `/search?${params.toString()}` : '/search';
    router.push(newURL, { scroll: false });
  }, [router]);

  const performSearch = useCallback(async (searchQuery: string, page: number = 1, volume: string = '') => {
    if (!searchQuery.trim()) {
      setSearchState(prev => ({
        ...prev,
        results: [],
        error: null
      }));
      setPaginationState(prev => ({
        ...prev,
        totalResults: 0
      }));
      updateURL('', 1, '');
      return;
    }

    setSearchState(prev => ({
      ...prev,
      loading: true,
      error: null
    }));

    try {
      const params = new URLSearchParams({
        q: searchQuery,
        page: page.toString(),
        per_page: perPage.toString(),
        ...(volume && { volume })
      });

      const response = await fetch(`/api/search?${params}`);
      const data: SearchResponse = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Search failed');
      }

      setSearchState(prev => ({
        ...prev,
        results: data.results,
        loading: false,
        error: null
      }));
      setPaginationState({
        currentPage: page,
        totalResults: data.found,
        searchTime: data.search_time_ms || data.text_match || 0
      });
      
      // Update URL with current search parameters
      updateURL(searchQuery, page, volume);
    } catch (err) {
      setSearchState(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'An error occurred',
        results: []
      }));
      setPaginationState(prev => ({
        ...prev,
        totalResults: 0
      }));
    }
  }, [updateURL]);

  // Debounce the search query
  useEffect(() => {
    const timer = setTimeout(() => {
      setSearchState(prev => ({
        ...prev,
        debouncedQuery: prev.query
      }));
    }, 1000); // 300ms delay

    return () => clearTimeout(timer);
  }, [searchState.query]);

  // Trigger search when debounced query changes (only after initialization)
  useEffect(() => {
    if (!uiState.isInitialized) return;
    
    // Check if this is the initial load by comparing with URL params
    const urlQuery = searchParams.get('q') || '';
    const isInitialLoad = searchState.debouncedQuery === urlQuery && searchState.query === urlQuery;
    
    if (searchState.debouncedQuery.trim()) {
      if (isInitialLoad) {
        // On initial load, use the current page from state (which was set from URL)
        performSearch(searchState.debouncedQuery, paginationState.currentPage, selectedVolume);
      } else {
        // On subsequent searches, reset to page 1
        setPaginationState(prev => ({ ...prev, currentPage: 1 }));
        performSearch(searchState.debouncedQuery, 1, selectedVolume);
      }
    } else {
      setSearchState(prev => ({
        ...prev,
        results: [],
        error: null
      }));
      setPaginationState(prev => ({
        ...prev,
        totalResults: 0
      }));
      updateURL('', 1, '');
    }
  }, [searchState.debouncedQuery, selectedVolume, performSearch, uiState.isInitialized, updateURL, searchParams, paginationState.currentPage, searchState.query]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setPaginationState(prev => ({ ...prev, currentPage: 1 }));
    performSearch(searchState.query, 1, selectedVolume);
  };

  const handlePageChange = (page: number) => {
    setPaginationState(prev => ({ ...prev, currentPage: page }));
    performSearch(searchState.debouncedQuery, page, selectedVolume);
  };

  const clearFilters = () => {
    setSelectedVolume('');
    if (searchState.query) {
      setPaginationState(prev => ({ ...prev, currentPage: 1 }));
      performSearch(searchState.query, 1, '');
    }
  };

  const getHeadingIcon = (type: string) => {
    switch (type) {
      case 'heading_major':
        return <BookOpen className="w-4 h-4" />;
      case 'heading_minor':
        return <FileText className="w-4 h-4" />;
      default:
        return <Hash className="w-4 h-4" />;
    }
  };

  const getHeadingStyle = (type: string) => {
    switch (type) {
      case 'heading_major':
        return 'text-lg font-bold text-gray-900';
      case 'heading_minor':
        return 'text-base font-semibold text-gray-800';
      default:
        return 'text-sm font-medium text-gray-700';
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

  const renderHighlightedText = (text: string) => {
    return <div dangerouslySetInnerHTML={{ __html: text }} className="text-sm text-gray-600" />;
  };

  // Memoized function to find the best match from all matched tokens
  const getBestMatchedToken = useMemo(() => {
    return (contentItems: ContentItem[]) => {
      if (!contentItems || contentItems.length === 0) return null;

      // Collect all matched tokens with their frequency and context
      const tokenFrequency = new Map<string, { count: number; contexts: string[] }>();
      
      contentItems.forEach(item => {
        if (item.matched_tokens && item.matched_tokens.length > 0) {
          item.matched_tokens.forEach(token => {
            const existing = tokenFrequency.get(token) || { count: 0, contexts: [] };
            existing.count += 1;
            existing.contexts.push(item.content_type);
            tokenFrequency.set(token, existing);
          });
        }
      });

      if (tokenFrequency.size === 0) return null;

      // Find the best match based on frequency and relevance
      let bestToken = '';
      let bestScore = 0;

      tokenFrequency.forEach((data, token) => {
        // Score based on frequency and token length (longer tokens are often more specific)
        const score = data.count * (token.length > 3 ? 2 : 1);
        if (score > bestScore) {
          bestScore = score;
          bestToken = token;
        }
      });

      return {
        token: bestToken,
        frequency: tokenFrequency.get(bestToken)?.count || 0,
        contexts: tokenFrequency.get(bestToken)?.contexts || []
      };
    };
  }, []);

  const volumes = [1, 2, 3, 4, 5, 6]; // Based on your data structure

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Search Islamic Rulings</h1>
          <p className="text-gray-600">Search through comprehensive Islamic fatwa collection</p>
        </div>

        {/* Search Form */}
        <Card className="mb-6">
          <CardContent className="p-6">
            <form onSubmit={handleSearch} className="space-y-4">
              <div className="flex gap-2">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                  <input
                    type="text"
                    value={searchState.query}
                    onChange={(e) => setSearchState(prev => ({ ...prev, query: e.target.value }))}
                    placeholder="Search for Islamic rulings, questions, or topics..."
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <Button type="submit" disabled={searchState.loading} className="px-6">
                  {searchState.loading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    'Search'
                  )}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setUiState(prev => ({ ...prev, showFilters: !prev.showFilters }))}
                  className="px-4"
                >
                  <Filter className="w-4 h-4" />
                </Button>
              </div>

              {/* Filters */}
              {uiState.showFilters && (
                <div className="border-t pt-4 space-y-3">
                  <div className="flex items-center gap-4">
                    <label className="text-sm font-medium text-gray-700">Volume:</label>
                    <select
                      value={selectedVolume}
                      onChange={(e) => {
                        const newVolume = e.target.value;
                        setSelectedVolume(newVolume);
                        if (searchState.query.trim()) {
                          setPaginationState(prev => ({ ...prev, currentPage: 1 }));
                          performSearch(searchState.query, 1, newVolume);
                        }
                      }}
                      className="border border-gray-300 rounded-md px-3 py-1 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="">All Volumes</option>
                      {volumes.map(vol => (
                        <option key={vol} value={vol.toString()}>Volume {vol}</option>
                      ))}
                    </select>
                    {selectedVolume && (
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={clearFilters}
                        className="text-gray-500 hover:text-gray-700"
                      >
                        <X className="w-4 h-4 mr-1" />
                        Clear
                      </Button>
                    )}
                  </div>
                </div>
              )}
            </form>
          </CardContent>
        </Card>

        {/* Search Results */}
        {searchState.error && (
          <Card className="mb-6 border-red-200 bg-red-50">
            <CardContent className="p-4">
              <div className="text-red-700">
                <strong>Error:</strong> {searchState.error}
              </div>
            </CardContent>
          </Card>
        )}

        {searchState.query && !searchState.loading && (
          <div className="mb-4 text-sm text-gray-600">
            {paginationState.totalResults > 0 ? (
              <>
                Found {paginationState.totalResults.toLocaleString()} results in {paginationState.searchTime}ms
                {selectedVolume && ` (filtered by Volume ${selectedVolume})`}
              </>
            ) : (
              'No results found'
            )}
          </div>
        )}

        {/* Results */}
        <div className="space-y-4">
          {searchState.results.map((hit, index) => {
            // Extract the first content item for display
            const firstContentItem = hit.document.content_items[0];
            const headingText = hit.document.heading?.text || firstContentItem?.value || 'Search Result';
            const headingType = hit.document.heading?.type || 'default';
            
            // Get the best matched token for this hit
            const bestMatch = getBestMatchedToken(hit.document.content_items);
            
            return (
              <Link key={`${index}-${hit.text_match}`} href={`/volume/${hit.document.vol}/section/${hit.document.section_id}`}>
                <Card className={`transition-all duration-200 cursor-pointer ${getCardStyle(headingType)}`}>
                  <CardHeader className="pb-3">
                    <div className="flex items-start gap-3">
                      <div className="mt-1 flex-shrink-0">
                        {getHeadingIcon(headingType)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <CardTitle className={`${getHeadingStyle(headingType)} line-clamp-2 mb-2`}>
                          {headingText}
                        </CardTitle>
                        <div className="flex items-center gap-2 mb-3">
                          {hit.document.vol && (
                            <Badge variant="outline" className="text-xs">
                              Volume {hit.document.vol}
                            </Badge>
                          )}
                          {hit.document.heading?.page_number && (
                            <Badge variant="outline" className="text-xs">
                              Page {hit.document.heading.page_number}
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>
                  </CardHeader>
                </Card>
              </Link>
            );
          })}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="mt-8 flex items-center justify-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handlePageChange(paginationState.currentPage - 1)}
              disabled={paginationState.currentPage === 1 || searchState.loading}
            >
              <ChevronLeft className="w-4 h-4" />
              Previous
            </Button>

            <div className="flex items-center gap-1">
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                let pageNum;
                if (totalPages <= 5) {
                  pageNum = i + 1;
                } else if (paginationState.currentPage <= 3) {
                  pageNum = i + 1;
                } else if (paginationState.currentPage >= totalPages - 2) {
                  pageNum = totalPages - 4 + i;
                } else {
                  pageNum = paginationState.currentPage - 2 + i;
                }

                return (
                  <Button
                    key={pageNum}
                    variant={paginationState.currentPage === pageNum ? "default" : "outline"}
                    size="sm"
                    onClick={() => handlePageChange(pageNum)}
                    disabled={searchState.loading}
                    className="w-8 h-8 p-0"
                  >
                    {pageNum}
                  </Button>
                );
              })}
            </div>

            <Button
              variant="outline"
              size="sm"
              onClick={() => handlePageChange(paginationState.currentPage + 1)}
              disabled={paginationState.currentPage === totalPages || searchState.loading}
            >
              Next
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
        )}

        {/* Empty state */}
        {!searchState.query && !searchState.loading && (
          <div className="text-center py-12">
            <Search className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Search Islamic Rulings</h3>
            <p className="text-gray-600 max-w-md mx-auto">
              Enter your search query above to find relevant Islamic rulings, fatwas, and religious guidance
              from our comprehensive collection.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
