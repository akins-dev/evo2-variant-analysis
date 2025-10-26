import type { GenomeAssemblyFromSearch, UCSCGenomeResponse } from "@/types/genomes";

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



