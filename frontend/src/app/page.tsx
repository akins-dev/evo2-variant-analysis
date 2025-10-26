"use client";

import GeneViewer from "@/components/gene-viewer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import {
  getAvailableGenomes,
  getGenomeChromosomes,
  searchGenes,
  type ChromosomeFromSearch,
  type GeneFromSearch,
  type GenomeAssemblyFromSearch,
} from "@/utils/genome-api";
import { Search } from "lucide-react";
import { useEffect, useState } from "react";

type modeType = "browse" | "search";

export default function HomePage() {
  const [genomes, setGenomes] = useState<GenomeAssemblyFromSearch[]>([]);
  const [selectedGenome, setSelectedGenome] = useState<string>("hg38");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chromosomes, setChromosomes] = useState<ChromosomeFromSearch[]>([]);
  const [selelctedChromosome, setSelectedChromosome] = useState<string>("chr1");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [mode, setMode] = useState<modeType>("search");
  const [searchResults, setSearchResults] = useState<GeneFromSearch[]>([]);
  const [selectedGene, setSelectedGene] = useState<GeneFromSearch | null>(null);

  useEffect(() => {
    const fetchGenomes = async () => {
      try {
        setIsLoading(true);
        const data = await getAvailableGenomes();

        if (data.genomes?.Human) {
          setGenomes(data.genomes.Human);
        }
      } catch (error) {
        if (error instanceof Error) {
          setError(`Failed to fetch genome assemblies. Error: ${error}`);
        } else {
          setError("Failed to fetch genome assemblies.");
        }
      } finally {
        setIsLoading(false);
      }
    };

    void fetchGenomes();
  }, []);

  useEffect(() => {
    const fetchChromosomes = async () => {
      try {
        setIsLoading(true);
        const data = await getGenomeChromosomes(selectedGenome);
        setChromosomes(data.chromosomes);
        console.log(data.chromosomes);
        if (data.chromosomes.length > 0) {
          setSelectedChromosome(data.chromosomes[0]!.name);
        }
      } catch (error) {
        console.log("fetchChromosomes error: ", error);
        setError("Failed to fetch chromosome data.");
      } finally {
        setIsLoading(false);
      }
    };

    void fetchChromosomes();
  }, [selectedGenome]);

  const performGeneSearch = async (
    query: string,
    genome: string,
    filterFn?: (gene: GeneFromSearch) => boolean,
  ) => {
    try {
      setIsLoading(true);
      const data: [string, string, GeneFromSearch[]] = await searchGenes(
        query,
        genome,
      );
      const results = filterFn ? data[2].filter(filterFn) : data[2];
      console.log("results", data[2]);

      setSearchResults(results);
    } catch (error) {
      console.log("error", error);
      setError("Failed to search genes.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!selelctedChromosome || mode !== "browse") return;
    void performGeneSearch(
      selelctedChromosome,
      selectedGenome,
      (gene: GeneFromSearch) => gene.chromosome === selelctedChromosome,
    );
  }, [selelctedChromosome, selectedGenome, mode]);

  const handleGenomeChange = (value: string) => {
    setSelectedGenome(value);
    setSearchResults([]);
    setSelectedGene(null);
  };

  const switchMode = (newMode: modeType) => {
    if (newMode === mode) return;

    setSearchResults([]);
    setSelectedGene(null);
    setError(null);

    if (newMode === "browse" && selelctedChromosome) {
      void performGeneSearch(
        selelctedChromosome,
        selectedGenome,
        (gene: GeneFromSearch) => gene.chromosome === selelctedChromosome,
      );
    }

    setMode(newMode);
  };

  const handleSearch = async (e: React.FormEvent | null = null) => {
    if (e) e.preventDefault();
    console.log("searchQuery", searchQuery);
    if (!searchQuery.trim()) return;

    // perform gene search
    void performGeneSearch(searchQuery, selectedGenome);
  };

  const loadBRCA1Example = () => {
    setMode("search");
    setSearchQuery("BRCA1");
    void performGeneSearch("BRCA1", selectedGenome);
  };


  console.log(selectedGene);

  return (
    <div className="bg-secondary min-h-screen">
      <header className="border-primary/10 border-b bg-white">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="relative">
              <h1 className="text-primary text-xl font-light tracking-wide">
                <span className="font-normal">EVO</span>
                <span className="text-accent">2</span>
              </h1>
              <div className="bg-accent absolute -bottom-1 left-0 h-1 w-12"></div>
            </div>
          </div>
          <span className="text-primary/70 text-sm font-light">
            Variant Analysis
          </span>
        </div>
      </header>

      <main className="container mx-auto px-6 py-6">
        {selectedGene ? (
          <GeneViewer gene={selectedGene} genomeId={selectedGenome} onClose={() => setSelectedGene(null)} />
        ) : (
          <>
            <Card className="mb-6 gap-0 border-none bg-white py-0 shadow-sm">
              <CardHeader className="pt-4 pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-primary/70 text-sm font-normal">
                    Genome Assembly
                  </CardTitle>
                  <div className="text-primary/60 text-xs">
                    Organism: <span className="font-medium">Human</span>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pb-4">
                <Select
                  value={selectedGenome}
                  onValueChange={handleGenomeChange}
                  disabled={isLoading}
                >
                  <SelectTrigger className="border-primary/10 h-9 w-full">
                    <SelectValue placeholder="Select genome assembly" />
                  </SelectTrigger>
                  <SelectContent>
                    {genomes?.map((genome) => (
                      <SelectItem key={genome.id} value={genome.id}>
                        {genome.id} - {genome.name}
                        {genome.active ? " (active)" : ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {selectedGenome && (
                  <p className="text-primary/60 mt-2 text-xs">
                    {
                      genomes.find((genome) => genome.id === selectedGenome)
                        ?.sourceName
                    }
                  </p>
                )}
              </CardContent>
            </Card>

            <Card className="mt-6 gap-0 border-none bg-white py-0 pb-4 shadow-sm">
              <CardHeader className="pt-4 pb-2">
                <CardTitle className="text-primary/70 text-sm font-normal">
                  Browse
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Tabs
                  value={mode}
                  onValueChange={(value) => switchMode(value as modeType)}
                >
                  <TabsList className="bg-secondary mb-4 h-12 sm:h-9">
                    <TabsTrigger
                      className="data-[state=active]:text-primary data-[state=active]:bg-white"
                      value="search"
                    >
                      Search Genes
                    </TabsTrigger>
                    <TabsTrigger
                      className="data-[state=active]:text-primary data-[state=active]:bg-white"
                      value="browse"
                    >
                      <p className="text-start">
                        Browse <br className="flex sm:hidden" />
                        Chromosomes
                      </p>
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="search" className="mt-0">
                    <div className="space-y-4">
                      <form
                        onSubmit={handleSearch}
                        className="flex flex-col gap-3 sm:flex-row"
                      >
                        <div className="relative flex-1">
                          <Input
                            type="text"
                            placeholder="Enter genes symbol or name..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="border-primary/10 h-9 pr-10"
                          />
                          <Button
                            className="bg-primary hoveer:bg-primary/90 absolute top-0 right-0 h-full cursor-pointer rounded-l-none text-white"
                            disabled={isLoading || !searchQuery.trim()}
                            size="icon"
                            type="submit"
                          >
                            <Search className="h-4 w-4" />
                            <span className="sr-only">Search</span>
                          </Button>
                        </div>
                      </form>
                      <Button
                        variant="link"
                        onClick={loadBRCA1Example}
                        className="text-accent hover:text-accent/80 h-auto cursor-pointer p-0"
                      >
                        Try BRCA1 example
                      </Button>
                    </div>
                  </TabsContent>
                  <TabsContent value="browse" className="mt-0">
                    <div className="max-h-[156px] overflow-y-auto pr-1">
                      <div className="flex flex-wrap gap-2">
                        {chromosomes.map((chr) => (
                          <Button
                            key={chr.name}
                            variant={
                              selelctedChromosome === chr.name
                                ? "default"
                                : "outline"
                            }
                            className={cn(
                              "border-bg-primary/10 h-8 cursor-pointer rounded-md px-3 text-sm",
                              selelctedChromosome !== chr.name &&
                                "hover:text-primary hover:bg-secondary",
                            )}
                            onClick={() => setSelectedChromosome(chr.name)}
                          >
                            {chr.name} ({chr.size.toLocaleString()} bp)
                          </Button>
                        ))}
                      </div>
                    </div>
                    {/* <div className="mt-4">
                  <Button className="bg-accent hover:bg-accent/90">bp: base pair</Button>
                </div> */}
                  </TabsContent>
                </Tabs>

                {isLoading && (
                  <div className="flex justify-center py-4">
                    <div className="border-primary/30 border-t-accent h-6 w-6 animate-spin rounded-full border-2" />
                  </div>
                )}

                {error && (
                  <div className="mt-4 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                    {error}
                  </div>
                )}

                {searchResults.length > 0 && !isLoading && (
                  <div className="mt-6">
                    <div className="mb-2">
                      <h4 className="text-primary/70 text-xs font-normal">
                        {mode === "search" ? (
                          <>
                            Search Results:{" "}
                            <span className="text-primary font-medium">
                              {searchResults.length} genes
                            </span>
                          </>
                        ) : (
                          <>
                            Genes on {selelctedChromosome}:{" "}
                            <span className="text-primary font-medium">
                              {searchResults.length} found
                            </span>
                          </>
                        )}
                      </h4>
                    </div>

                    <div className="border-primary/5 overflow-hidden rounded-md border">
                      <Table>
                        <TableHeader>
                          <TableRow className="bg-secondary/50 hover:bg-secondary/70">
                            <TableHead className="text-primary/70 text-xs font-normal">
                              Symbol
                            </TableHead>
                            <TableHead className="text-primary/70 text-xs font-normal">
                              Name
                            </TableHead>
                            <TableHead className="text-primary/70 text-xs font-normal">
                              Location
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {searchResults.map((gene, index) => (
                            <TableRow
                              key={`${gene.symbol}-${index}`}
                              className="border-primary/5 hhover:bg-secondary/50 cursor-pointer border-b"
                              onClick={() => setSelectedGene(gene)}
                            >
                              <TableCell className="text-primary py-2 font-medium">
                                {gene.symbol}
                              </TableCell>
                              <TableCell className="text-primary py-2 font-medium">
                                {gene.name}
                              </TableCell>
                              <TableCell className="text-primary py-2 font-medium">
                                {gene.chromosome}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                )}

                {!isLoading && !error && searchResults.length === 0 && (
                  <div className="flex h-48 flex-col items-center justify-center text-center text-gray-400">
                    <Search className="mb-4 h-10 w-10 text-gray-400" />{" "}
                    <p className="text-sm leading-relaxed">
                      {mode === "search"
                        ? "Enter a gene or symbol and click search"
                        : selelctedChromosome
                          ? "No gene found on this chromosome"
                          : "Select a chromosome to view genes"}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </>
        )}
      </main>
    </div>
  );
}
