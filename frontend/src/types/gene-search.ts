export interface GeneFromSearch {
  symbol: string;
  name: string;
  chromosome: string;
  description: string;
  geneId: string;
}

export interface GeneDetails {
  GeneID: string[];
  GenomicInfo: (string | null)[];
  Symbol: string[];
  chromosome: string[];
  description: string[];
  map_location: string[];
  type_of_gene: string[];
}

export interface GeneApiResponse
  extends Array<number | string[] | GeneDetails | string[][]> {
  0: number; // numerical identifier
  1: string[]; // list of gene IDs or accession IDs
  2: GeneDetails; // main details object
  3: string[]; // tabular data per gene
  4: string[]; // annotation source labels
}
