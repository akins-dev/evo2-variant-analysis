import type { GeneApiResponse, GeneFromSearch } from "@/types/gene-search";

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
          console.log("searchGenes[ERROR]: Something went wrong");
          continue;
        }
      }
    }
  }

  return [query, genome, results];
}
