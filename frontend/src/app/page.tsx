"use client";

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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import {
  getAvailableGenomes,
  getGenomeChromosomes,
  type ChromosomeFromSearch,
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
  const [selelctedChromosomes, setSelectedChromosomes] =
    useState<string>("chr1");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [mode, setMode] = useState<modeType>("search");

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
          setError(
            "Failed to fetch genome assemblies.",
          );
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
          setSelectedChromosomes(data.chromosomes[0]!.name);
        }
      } catch (error) {
        setError("Failed to fetch chromosome data.");
      } finally {
        setIsLoading(false);
      }
    };

    void fetchChromosomes();
  }, [selectedGenome]);

  const handleGenomeChange = (value: string) => {
    setSelectedGenome(value);
  };

  const switchMode = (newMode: modeType) => {
    if (newMode === mode) return;
    setMode(newMode);
  };

  const handleSearch = async (e: React.FormEvent | null = null) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) return;

    // perform gene search
  };

  const loadBRCA1Example = () => {
    setMode("search");
    setSearchQuery("BRCA1");

    // handle search
    void handleSearch();
  };

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
              <TabsList className="bg-secondary mb-4">
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
                  Browse Chromosomes
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
                          selelctedChromosomes === chr.name
                            ? "default"
                            : "outline"
                        }
                        className={cn(
                          "border-bg-primary/10 h-8 cursor-pointer rounded-md px-3 text-sm",
                          selelctedChromosomes !== chr.name &&
                            "hover:text-primary hover:bg-secondary",
                        )}
                        onClick={() => setSelectedChromosomes(chr.name)}
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
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
