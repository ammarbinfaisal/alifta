import { NextRequest, NextResponse } from 'next/server';
import Typesense from 'typesense';

const client = new Typesense.Client({
  nodes: [{
    host: process.env.TYPSENSE_HOST!,
    port: parseInt(process.env.TYPSENSE_PORT!),
    protocol: "https"
  }],
  apiKey: process.env.TYPESENSE_API_KEY!,
  connectionTimeoutSeconds: 5
});

// Common Islamic terms and their variations for spell correction
const islamicTermsMap: Record<string, string[]> = {
  'prayer': ['salah', 'salat', 'namaz', 'dua'],
  'fasting': ['sawm', 'roza', 'fast'],
  'pilgrimage': ['hajj', 'umrah'],
  'charity': ['zakat', 'sadaqah', 'zakah'],
  'ablution': ['wudu', 'ghusl', 'tayammum'],
  'mosque': ['masjid', 'masjed'],
  'quran': ['quraan', 'koran', 'quran'],
  'prophet': ['rasul', 'messenger', 'nabi'],
  'marriage': ['nikah', 'wedding'],
  'divorce': ['talaq', 'khula'],
  'inheritance': ['mirath', 'warith'],
  'forbidden': ['haram', 'haraam'],
  'permitted': ['halal', 'halaal'],
  'jurisprudence': ['fiqh', 'fikh'],
  'scholar': ['alim', 'mufti', 'sheikh'],
  'ruling': ['fatwa', 'hukm'],
  'worship': ['ibadah', 'ibadat'],
  'faith': ['iman', 'aqeedah'],
  'sin': ['gunah', 'ithm'],
  'repentance': ['tawbah', 'taubah']
};

// Function to expand query with Islamic term variations
function expandQuery(query: string): string {
  let expandedQuery = query.toLowerCase();
  
  // Check for Islamic terms and add variations
  for (const [english, arabic] of Object.entries(islamicTermsMap)) {
    const regex = new RegExp(`\\b${english}\\b`, 'gi');
    if (regex.test(expandedQuery)) {
      // Add Arabic/alternative terms as OR conditions
      const alternatives = arabic.join(' OR ');
      expandedQuery = expandedQuery.replace(regex, `(${english} OR ${alternatives})`);
    }
    
    // Also check if any Arabic terms are used and suggest English
    for (const term of arabic) {
      const arabicRegex = new RegExp(`\\b${term}\\b`, 'gi');
      if (arabicRegex.test(expandedQuery)) {
        expandedQuery = expandedQuery.replace(arabicRegex, `(${term} OR ${english})`);
      }
    }
  }
  
  return expandedQuery;
}

// Function to get search suggestions
async function getSearchSuggestions(query: string): Promise<string[]> {
  try {
    // Get suggestions from Typesense
    const suggestions = await client.collections('islamic_rulings').documents().search({
      q: query,
      query_by: 'heading.text,content_items.text',
      per_page: 5,
      num_typos: 3,
      typo_tokens_threshold: 1,
      prefix: true,
      drop_tokens_threshold: 2
    });

    // Extract unique terms from results for suggestions
    const suggestionSet = new Set<string>();
    
    suggestions.hits?.forEach((hit: any) => {
      const heading = hit.document?.heading?.text || '';
      const words = heading.toLowerCase().split(/\s+/).filter((word: string) => 
        word.length > 3 && word.includes(query.toLowerCase().substring(0, 3))
      );
      words.forEach((word: string) => suggestionSet.add(word));
    });

    return Array.from(suggestionSet).slice(0, 5);
  } catch (error) {
    console.error('Suggestion error:', error);
    return [];
  }
}

// Function to detect and correct common spelling mistakes
function correctSpelling(query: string): { corrected: string; wasCorrected: boolean } {
  const corrections: Record<string, string> = {
    // Common misspellings
    'prayar': 'prayer',
    'prayr': 'prayer',
    'salaat': 'salah',
    'salaah': 'salah',
    'zakaat': 'zakat',
    'zakaah': 'zakat',
    'hajjj': 'hajj',
    'umraa': 'umrah',
    'masjeed': 'masjid',
    'masdjid': 'masjid',
    'quaran': 'quran',
    'koran': 'quran',
    'nikkaah': 'nikah',
    'talaaq': 'talaq',
    'halaal': 'halal',
    'haraam': 'haram',
    'fiqah': 'fiqh',
    'aqeeda': 'aqeedah',
    'tawba': 'tawbah'
  };

  let corrected = query.toLowerCase();
  let wasCorrected = false;

  for (const [wrong, right] of Object.entries(corrections)) {
    const regex = new RegExp(`\\b${wrong}\\b`, 'gi');
    if (regex.test(corrected)) {
      corrected = corrected.replace(regex, right);
      wasCorrected = true;
    }
  }

  return { corrected, wasCorrected };
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const originalQuery = searchParams.get('q');
  const page = parseInt(searchParams.get('page') || '1');
  const perPage = parseInt(searchParams.get('per_page') || '10');
  const volume = searchParams.get('volume');
  const getSuggestions = searchParams.get('suggestions') === 'true';

  if (!originalQuery) {
    return NextResponse.json({ error: 'Query parameter is required' }, { status: 400 });
  }

  // If only suggestions are requested
  if (getSuggestions) {
    const suggestions = await getSearchSuggestions(originalQuery);
    return NextResponse.json({ suggestions });
  }

  try {
    // Step 1: Correct spelling
    const { corrected: spellingCorrected, wasCorrected } = correctSpelling(originalQuery);
    
    // Step 2: Expand query with Islamic term variations
    const expandedQuery = expandQuery(spellingCorrected);
    
    // Step 3: Perform primary search with enhanced parameters
    const searchParameters = {
      q: expandedQuery,
      query_by: 'heading.text,content_items.text,content_items',
      page: page,
      per_page: perPage,
      highlight_full_fields: 'heading.text,content_items.text,content_items',
      snippet_threshold: 30,
      num_typos: 3, // Increased typo tolerance
      typo_tokens_threshold: 1,
      drop_tokens_threshold: 2, // Allow dropping more tokens for better semantic matching
      prefix: true, // Enable prefix matching
      ...(volume && { filter_by: `vol:=${volume}` })
    };

    const searchResults = await client.collections('islamic_rulings').documents().search(searchParameters);

    // Step 4: If no results and query was corrected, try original query
    let fallbackResults = null;
    if (searchResults.found === 0 && wasCorrected) {
      const fallbackParameters = {
        q: originalQuery,
        query_by: 'heading.text,content_items.text',
        page: page,
        per_page: perPage,
        highlight_full_fields: 'heading.text,content_items.text',
        snippet_threshold: 30,
        num_typos: 4, // Even more lenient for fallback
        typo_tokens_threshold: 1,
        drop_tokens_threshold: 3,
        prefix: true,
        ...(volume && { filter_by: `vol:=${volume}` })
      };
      
      try {
        fallbackResults = await client.collections('islamic_rulings').documents().search(fallbackParameters);
      } catch (fallbackError) {
        console.error('Fallback search error:', fallbackError);
      }
    }

    // Step 5: Get suggestions for better search experience
    const suggestions = searchResults.found < 5 ? await getSearchSuggestions(originalQuery) : [];

    const finalResults = fallbackResults && fallbackResults.found > 0 ? fallbackResults : searchResults;

    return NextResponse.json({
      results: finalResults.hits,
      found: finalResults.found,
      page: finalResults.page,
      search_time_ms: finalResults.search_time_ms,
      facet_counts: finalResults.facet_counts,
      // Enhanced response data
      query_info: {
        original_query: originalQuery,
        processed_query: expandedQuery,
        spelling_corrected: wasCorrected,
        corrected_query: wasCorrected ? spellingCorrected : null
      },
      suggestions: suggestions,
      used_fallback: fallbackResults && fallbackResults.found > 0
    });

  } catch (error) {
    console.error('Search error:', error);
    
    // Enhanced error handling with suggestions
    const suggestions = await getSearchSuggestions(originalQuery);
    
    return NextResponse.json(
      { 
        error: 'Search failed', 
        details: error instanceof Error ? error.message : 'Unknown error',
        suggestions: suggestions,
        query_info: {
          original_query: originalQuery
        }
      },
      { status: 500 }
    );
  }
}
