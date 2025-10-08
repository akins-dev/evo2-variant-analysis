export interface GenomeAssemblyFromSearch {
  id: string;
  name: string;
  sourceName: string;
  active: boolean;
}

interface UCSCGenome {
  description: string;
  nibPath: string;
  organism: string;
  defaultPos: string;
  active: number;
  orderKey: number;
  genome: string;
  scientificName: string;
  htmlPath: string;
  hgNearOk: number;
  hgPbOk: number;
  sourceName: string;
  taxId: number;
}

interface UCSCGenomeResponse {
  downloadTime: string;
  downloadTimeStamp: number;
  dataTime: string;
  dataTimeStamp: number;
  ucscGenomes: Record<string, UCSCGenome>;
}

interface UCSChromosomeResponse {
  downloadTime: string;
  downloadTimeStamp: number;
  genome: string;
  dataTime: string;
  dataTimeStamp: number;
  chromCount: number;
  chromosomes: Record<string, number>;
}

export interface ChromosomeFromSearch {
  name: string;
  size: number;
}

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

export async function getAvailableGenomes() {
  const apiUrl = "https://api.genome.ucsc.edu/list/ucscGenomes";
  const response = await fetch(apiUrl);
  if (!response.ok) {
    throw new Error("Failed to fetch genome data from UCSC API");
  }

  const genomeData = (await response.json()) as UCSCGenomeResponse;

  if (!genomeData.ucscGenomes) {
    throw new Error("UCSC API Error: missing ucscGenomes");
  }

  const genomes = genomeData.ucscGenomes;
  const structuresGenomes: Record<string, GenomeAssemblyFromSearch[]> = {};

  for (const genomeId in genomes) {
    const genomeInfo = genomes[genomeId];
    const organism = genomeInfo?.organism ?? "Other";

    structuresGenomes[organism] ??= [];

    structuresGenomes[organism].push({
      id: genomeId,
      name: genomeInfo?.description ?? genomeId,
      sourceName: genomeInfo?.sourceName ?? genomeId,
      active: !!genomeInfo?.active,
    });
  }

  return { genomes: structuresGenomes };
}

export async function getGenomeChromosomes(genomeId: string) {
  const apiUrl = `https://api.genome.ucsc.edu/list/chromosomes?genome=${genomeId}`;
  const response = await fetch(apiUrl);
  if (!response.ok) {
    throw new Error("Failed to fetch chromosome list from UCSC API");
  }

  const chromosomeData = (await response.json()) as UCSChromosomeResponse;

  if (!chromosomeData.chromosomes) {
    throw new Error("UCSC API Error: missing chromosomes");
  }

  const chromosomes: ChromosomeFromSearch[] = [];

  for (const chromId in chromosomeData.chromosomes) {
    if (
      chromId.includes("_") ||
      chromId.includes("Un") ||
      chromId.includes("random")
    )
      continue;

    chromosomes.push({
      name: chromId,
      size: chromosomeData.chromosomes[chromId]!,
    });
  }

  // Sort chromosomes in natural order (1, 2, ..., 10, 11, ..., X, Y)
  chromosomes.sort((a, b) => {
    const numA = a.name.replace("chr", "");
    const numB = b.name.replace("chr", "");

    const isNumA = /^\d+$/.test(numA);
    const isNumB = /^\d+$/.test(numB);

    if (isNumA && isNumB) return Number(numA) - Number(numB);
    if (isNumA) return -1;
    if (isNumB) return 1;
    return numA.localeCompare(numB);
  });

  return { chromosomes };
}

export async function searchGenes(
  query: string,
  genome: string,
): Promise<[string, string, GeneFromSearch[]]> {
  const url = "https://clinicaltables.nlm.nih.gov/api/ncbi_genes/v3/search";

  const params = new URLSearchParams({
    terms: query,
    df: "chromosome,Symbol,description,map_location,type_of_gene",
    ef: "chromosome,Symbol,description,map_location,type_of_gene,GenomicInfo,GeneID",
  });

  const response = await fetch(`${url}?${params}`);

  if (!response.ok) {
    throw new Error("Failed to fetch gene data from NCBI API");
  }

  const data = (await response.json()) as GeneApiResponse;
  const results: GeneFromSearch[] = [];

  if (data[0] > 0) {
    const fieldMap = data[2];
    const geneIds = fieldMap.GeneID || [];

    for (let i = 0; i < Math.min(10, data[0]); ++i) {
      if (i < data[3].length) {
        try {
          const display = data[3][i]!;

          let chrom = display[0];
          if (chrom && !chrom.startsWith("chr")) {
            chrom = `chr${chrom}`;
          }

          results.push({
            symbol: display[2]!,
            name: display[3]!,
            chromosome: chrom!,
            description: display[3]!,
            geneId: geneIds[i] ?? "",
          });
        } catch {
          continue;
        }
      }
    }
  }

  return [query, genome, results];
}
