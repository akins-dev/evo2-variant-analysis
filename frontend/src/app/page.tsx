"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  getAvailableGenomes,
  getGenomeChromosomes,
  type ChromosomeFromSearch,
  type GenomeAssemblyFromSearch,
} from "@/utils/genome-api";
import { useEffect, useState } from "react";

export default function HomePage() {
  const [genomes, setGenomes] = useState<GenomeAssemblyFromSearch[]>([]);
  const [selectedGenome, setSelectedGenome] = useState<string>("hg38");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chromosomes, setChromosomes] = useState<ChromosomeFromSearch[]>([]);
  const [selelctedChromosomes, setSelectedChromosomes] = useState<string>("chr1");

  useEffect(() => {
    const fetchGenomes = async () => {
      try {
        setIsLoading(true);
        const data = await getAvailableGenomes();

        if (data.genomes?.Human) {
          setGenomes(data.genomes.Human);
        }
      } catch (error) {
        setError("Failed to fetch genome assemblies.");
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

  return (
    <div className="min-h-screen bg-[#e9eeea]">
      <header className="border-primary/10 border-b bg-white">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="relative">
              <h1 className="text-primary text-xl font-light tracking-wide">
                <span className="font-normal">EVO</span>
                <span className="text-[#de8246]">2</span>
              </h1>
              <div className="absolute -bottom-1 left-0 h-1 w-12 bg-[#de8246]"></div>
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
      </main>
    </div>
  );
}
