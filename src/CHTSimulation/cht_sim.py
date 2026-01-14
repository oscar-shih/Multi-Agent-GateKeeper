# cht_sim.py
#
# Contract assumptions:
# - folder = dict
# - file   = list: ["TypeName", {kwargs...}]
#
# Scope:
# - 4 OpenFOAM CHT core files modeled:
#     1) system/controlDict (global)
#     2) system/<region>/fvSchemes (per-region, hardcoded preset "robust/upwind")
#     3) system/<region>/fvSolution (per-region, inject solver stanzas for T and p_rgh)
#     4) constant/<region>/thermophysicalProperties (per-region)
# - Time/0 per-region initial fields are included and defaults are injected:
#     - T default: 298.15 K (25°C)
#     - p_rgh default: 0.0 Pa (since absolute pressure reference p=1 atm is desired)
# - Absolute pressure reference is tracked as pressureReference_Pa (default 101325 Pa)
#
# Notes:
# - This models only internalField + dimensions for 0/* fields (boundaryField omitted for MVP).
# - Writer is expected to use pressureReference_Pa to set absolute pressure reference (p)
#   via solver-specific reference mechanism (e.g., pRefCell/pRefValue, boundary, etc.).
#
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class FileNode(RootModel[List[Any]]):
    root: List[Any]

    def type_tag(self) -> str:
        return self.root[0]

    def kwargs_dict(self) -> Dict[str, Any]:
        return self.root[1]

def _assert_node_shape(node: Any, tag: str) -> Dict[str, Any]:
    if not isinstance(node, list) or len(node) != 2:
        raise ValueError(f"{tag} node must be a JSON array of length 2: ['{tag}', {{...}}]")
    if node[0] != tag:
        raise ValueError(f"{tag} node tag must be '{tag}'")
    if not isinstance(node[1], dict):
        raise ValueError(f"{tag} node second element must be an object (kwargs)")
    return node[1]

class ControlDict(StrictModel):
    solverName: str = Field(..., description="OpenFOAM application/solver name, e.g. chtMultiRegionFoam")
    startTime: float = 0.0
    endTime: float
    deltaT: float

    writeControl: Literal["timeStep", "runTime", "adjustableRunTime", "clockTime", "cpuTime"] = "timeStep"
    writeInterval: float

    startFrom: Literal["startTime", "latestTime"] = "startTime"
    stopAt: Literal["endTime", "noWriteNow", "writeNow", "nextWrite"] = "endTime"
    writeFormat: Literal["ascii", "binary"] = "ascii"

    def check(self) -> None:
        errors = []
        if not self.solverName:
            errors.append("controlDict: solverName must be non-empty.")
        if self.endTime <= self.startTime:
            errors.append(f"controlDict: endTime ({self.endTime}) must be > startTime ({self.startTime}).")
        if self.deltaT <= 0:
            errors.append(f"controlDict: deltaT ({self.deltaT}) must be > 0.")
        if self.writeInterval <= 0:
            errors.append(f"controlDict: writeInterval ({self.writeInterval}) must be > 0.")
        
        if errors:
            raise ValueError("\n".join(errors))

class ControlDictNode(FileNode):
    @model_validator(mode="after")
    def _validate(self):
        _assert_node_shape(self.root, "controlDict")
        ControlDict.model_validate(self.root[1])
        return self

    def to_model(self) -> ControlDict:
        return ControlDict.model_validate(self.root[1])

class FvSchemes(StrictModel):
    """
    MVP: do NOT let the LLM specify detailed schemes.
    We hardcode to a known preset that the writer expands deterministically.
    """
    preset: Literal["robust/upwind"] = "robust/upwind"

    def check(self) -> None:
        if self.preset != "robust/upwind":
            raise ValueError("fvSchemes: only preset 'robust/upwind' is allowed in MVP.")

class FvSchemesNode(FileNode):
    @model_validator(mode="after")
    def _validate(self):
        kwargs = _assert_node_shape(self.root, "fvSchemes")
        FvSchemes.model_validate(kwargs or {})
        return self

    def to_model(self) -> FvSchemes:
        return FvSchemes.model_validate(self.root[1] or {})

class LinearSolverSettings(StrictModel):
    solver: str
    tolerance: float = 1e-7
    relTol: float = 0.0

    preconditioner: Optional[str] = None
    smoother: Optional[str] = None
    nPreSweeps: Optional[int] = Field(default=None, ge=0)
    nPostSweeps: Optional[int] = Field(default=None, ge=0)

class FvSolution(StrictModel):
    """
    MVP: ensure solvers contain required stanzas so the case doesn't crash:
      - fluid region: T and p_rgh
      - solid region: T
    """
    solvers: Dict[str, LinearSolverSettings] = Field(default_factory=dict)

    SIMPLE: Optional[Dict[str, Any]] = None
    PIMPLE: Optional[Dict[str, Any]] = None

    def inject_required_solvers(self, region_kind: Literal["fluid", "solid"]) -> None:
        if "T" not in self.solvers:
            self.solvers["T"] = LinearSolverSettings(
                solver="PBiCGStab",
                preconditioner="DILU",
                tolerance=1e-7,
                relTol=0.0,
            )
        if region_kind == "fluid" and "p_rgh" not in self.solvers:
            self.solvers["p_rgh"] = LinearSolverSettings(
                solver="PCG",
                preconditioner="DIC",
                tolerance=1e-7,
                relTol=0.0,
            )

    def check(self, region_kind: Literal["fluid", "solid"]) -> None:
        if "T" not in self.solvers:
            raise ValueError("fvSolution: missing required solver for T.")
        if region_kind == "fluid" and "p_rgh" not in self.solvers:
            raise ValueError("fvSolution: missing required solver for p_rgh in fluid region.")


class FvSolutionNode(FileNode):
    @model_validator(mode="after")
    def _validate(self):
        kwargs = _assert_node_shape(self.root, "fvSolution")
        FvSolution.model_validate(kwargs or {})
        return self

    def to_model(self) -> FvSolution:
        return FvSolution.model_validate(self.root[1] or {})

class ThermoType(StrictModel):
    type: str
    mixture: str
    transport: str
    thermo: str
    equationOfState: str
    specie: str
    energy: str

class SpecieBlock(StrictModel):
    nMoles: float = 1.0
    molWeight: float

class TransportConstBlock(StrictModel):
    mu: float
    Pr: float

class TransportGenericBlock(StrictModel):
    params: Dict[str, Any] = Field(default_factory=dict)

TransportBlock = Union[TransportConstBlock, TransportGenericBlock]

class ThermodynamicsBlock(StrictModel):
    coeffs: Dict[str, Any] = Field(default_factory=dict)

class MixtureBlock(StrictModel):
    specie: SpecieBlock
    thermodynamics: ThermodynamicsBlock
    transport: TransportBlock

class ThermophysicalProperties(StrictModel):
    thermoType: ThermoType
    mixture: MixtureBlock

    @model_validator(mode="after")
    def _enforce_transport(self):
        if self.thermoType.transport == "const":
            if not isinstance(self.mixture.transport, TransportConstBlock):
                raise ValueError("thermophysicalProperties: transport=const requires mixture.transport(mu, Pr).")
        return self

    def check(self) -> None:
        errors = []
        if not self.thermoType.type:
            errors.append("thermophysicalProperties: thermoType.type must be non-empty.")
        if self.mixture.specie.molWeight <= 0:
            errors.append(f"thermophysicalProperties: molWeight ({self.mixture.specie.molWeight}) must be > 0.")
        
        if isinstance(self.mixture.transport, TransportConstBlock):
             if self.mixture.transport.mu <= 0:
                 errors.append(f"thermophysicalProperties: mu ({self.mixture.transport.mu}) must be > 0.")
             if self.mixture.transport.Pr <= 0:
                 errors.append(f"thermophysicalProperties: Pr ({self.mixture.transport.Pr}) must be > 0.")

        if errors:
            raise ValueError("\n".join(errors))


class ThermophysicalPropertiesNode(FileNode):
    @model_validator(mode="after")
    def _validate(self):
        _assert_node_shape(self.root, "thermophysicalProperties")
        ThermophysicalProperties.model_validate(self.root[1])
        return self

    def to_model(self) -> ThermophysicalProperties:
        return ThermophysicalProperties.model_validate(self.root[1])

class VolScalarField(StrictModel):
    dimensions: List[int] = Field(..., min_length=7, max_length=7)
    internalField: float


class VolScalarFieldNode(FileNode):
    @model_validator(mode="after")
    def _validate(self):
        _assert_node_shape(self.root, "volScalarField")
        VolScalarField.model_validate(self.root[1])
        return self

    def to_model(self) -> VolScalarField:
        return VolScalarField.model_validate(self.root[1])

class RegionSystemConfig(StrictModel):
    fvSchemes: List[Any]
    fvSolution: List[Any]
    parsed_fvSchemes: FvSchemes = Field(default=None, exclude=True)
    parsed_fvSolution: FvSolution = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _parse(self):
        object.__setattr__(self, 'parsed_fvSchemes', FvSchemesNode.model_validate(self.fvSchemes).to_model())
        object.__setattr__(self, 'parsed_fvSolution', FvSolutionNode.model_validate(self.fvSolution).to_model())
        return self


class RegionConstantConfig(StrictModel):
    thermophysicalProperties: List[Any]

    parsed_thermo: ThermophysicalProperties = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _parse(self):
        object.__setattr__(self, 'parsed_thermo', ThermophysicalPropertiesNode.model_validate(self.thermophysicalProperties).to_model())
        return self


class RegionTimeFields(StrictModel):
    """
    Folder representing 0/<region>/.
    Fields maps fieldName -> ["volScalarField", {...}].
    """
    Fields: Dict[str, List[Any]] = Field(default_factory=dict)

    parsed_fields: Dict[str, VolScalarField] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def _parse(self):
        parsed: Dict[str, VolScalarField] = {}
        for name, node in self.Fields.items():
            if not isinstance(node, list) or len(node) != 2 or node[0] != "volScalarField":
                raise ValueError(f"Time field '{name}' must be ['volScalarField', {{...}}]")
            parsed[name] = VolScalarFieldNode.model_validate(node).to_model()
        object.__setattr__(self, 'parsed_fields', parsed)
        return self

class CHTSimulation(StrictModel):
    """
    LLM output contract:
    {
      "System": {
        "controlDict": ["controlDict", {...}],
        "Regions": {
          "fluids": { "<fluidRegion>": { "fvSchemes": [...], "fvSolution": [...] }, ... },
          "solids": { "<solidRegion>": { "fvSchemes": [...], "fvSolution": [...] }, ... }
        }
      },
      "Constant": {
        "Regions": {
          "fluids": { "<fluidRegion>": { "thermophysicalProperties": [...] }, ... },
          "solids": { "<solidRegion>": { "thermophysicalProperties": [...] }, ... }
        }
      },
      "Time": {
        "name": "0",
        "Regions": {
          "fluids": { "<fluidRegion>": { "Fields": { "T": [...], "p_rgh": [...] } }, ... },
          "solids": { "<solidRegion>": { "Fields": { "T": [...] } }, ... }
        }
      }
    }

    Defaults injected if missing:
      - 0/<region>/T = 298.15 K
      - 0/<fluidRegion>/p_rgh = 0.0 Pa
      - fvSolution solvers stanzas for T (all) and p_rgh (fluids)
    """

    System: Dict[str, Any]
    Constant: Dict[str, Any]
    Time: Optional[Dict[str, Any]] = None

    defaultTemperature_K: float = 298.15
    pressureReference_Pa: float = 101325.0
    default_p_rgh_Pa: float = 0.0

    parsed_controlDict: ControlDict = Field(default=None, exclude=True)

    parsed_system_fluids: Dict[str, RegionSystemConfig] = Field(default_factory=dict, exclude=True)
    parsed_system_solids: Dict[str, RegionSystemConfig] = Field(default_factory=dict, exclude=True)

    parsed_const_fluids: Dict[str, RegionConstantConfig] = Field(default_factory=dict, exclude=True)
    parsed_const_solids: Dict[str, RegionConstantConfig] = Field(default_factory=dict, exclude=True)

    parsed_time_fluids: Dict[str, RegionTimeFields] = Field(default_factory=dict, exclude=True)
    parsed_time_solids: Dict[str, RegionTimeFields] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def _parse_and_inject(self):
        if "controlDict" not in self.System:
            raise ValueError("System must include controlDict node.")
        object.__setattr__(self, 'parsed_controlDict', ControlDictNode.model_validate(self.System["controlDict"]).to_model())

        sys_regions = self.System.get("Regions")
        if not isinstance(sys_regions, dict):
            raise ValueError("System must include Regions folder (dict).")

        fluids = sys_regions.get("fluids", {})
        solids = sys_regions.get("solids", {})
        if not isinstance(fluids, dict) or not isinstance(solids, dict):
            raise ValueError("System.Regions.fluids and System.Regions.solids must be dicts.")

        object.__setattr__(self, 'parsed_system_fluids', {name: RegionSystemConfig.model_validate(cfg) for name, cfg in fluids.items()})
        object.__setattr__(self, 'parsed_system_solids', {name: RegionSystemConfig.model_validate(cfg) for name, cfg in solids.items()})

        if len(self.parsed_system_fluids) + len(self.parsed_system_solids) == 0:
            raise ValueError("At least one region (fluid or solid) must be provided.")

        for _, cfg in self.parsed_system_fluids.items():
            cfg.parsed_fvSolution.inject_required_solvers("fluid")
        for _, cfg in self.parsed_system_solids.items():
            cfg.parsed_fvSolution.inject_required_solvers("solid")

        const_regions = self.Constant.get("Regions")
        if not isinstance(const_regions, dict):
            raise ValueError("Constant must include Regions folder (dict).")

        cfluids = const_regions.get("fluids", {})
        csolids = const_regions.get("solids", {})
        if not isinstance(cfluids, dict) or not isinstance(csolids, dict):
            raise ValueError("Constant.Regions.fluids and Constant.Regions.solids must be dicts.")

        object.__setattr__(self, 'parsed_const_fluids', {name: RegionConstantConfig.model_validate(cfg) for name, cfg in cfluids.items()})
        object.__setattr__(self, 'parsed_const_solids', {name: RegionConstantConfig.model_validate(cfg) for name, cfg in csolids.items()})

        sys_fluid_names = set(self.parsed_system_fluids.keys())
        sys_solid_names = set(self.parsed_system_solids.keys())
        const_fluid_names = set(self.parsed_const_fluids.keys())
        const_solid_names = set(self.parsed_const_solids.keys())

        if sys_fluid_names != const_fluid_names:
            raise ValueError(
                f"Fluid regions mismatch between System and Constant: {sorted(sys_fluid_names)} vs {sorted(const_fluid_names)}"
            )
        if sys_solid_names != const_solid_names:
            raise ValueError(
                f"Solid regions mismatch between System and Constant: {sorted(sys_solid_names)} vs {sorted(const_solid_names)}"
            )

        if self.Time is None:
            self.Time = {"name": "0", "Regions": {"fluids": {}, "solids": {}}}

        if not isinstance(self.Time, dict):
            raise ValueError("Time must be an object if provided.")

        time_name = self.Time.get("name", "0")
        if not isinstance(time_name, str) or not time_name:
            raise ValueError("Time.name must be a non-empty string (e.g., '0').")
        self.Time["name"] = time_name

        time_regions = self.Time.get("Regions")
        if not isinstance(time_regions, dict):
            self.Time["Regions"] = {"fluids": {}, "solids": {}}
            time_regions = self.Time["Regions"]

        tfluids = time_regions.get("fluids", {})
        tsolids = time_regions.get("solids", {})
        if not isinstance(tfluids, dict) or not isinstance(tsolids, dict):
            raise ValueError("Time.Regions.fluids and Time.Regions.solids must be dicts.")

        object.__setattr__(self, 'parsed_time_fluids', {name: RegionTimeFields.model_validate(cfg) for name, cfg in tfluids.items()})
        object.__setattr__(self, 'parsed_time_solids', {name: RegionTimeFields.model_validate(cfg) for name, cfg in tsolids.items()})

        for r in sys_fluid_names:
            if r not in self.parsed_time_fluids:
                self.parsed_time_fluids[r] = RegionTimeFields(Fields={})
        for r in sys_solid_names:
            if r not in self.parsed_time_solids:
                self.parsed_time_solids[r] = RegionTimeFields(Fields={})

        T_dims = [0, 0, 0, 1, 0, 0, 0]
        p_dims = [1, -1, -2, 0, 0, 0, 0]

        for _, tf in self.parsed_time_fluids.items():
            if "T" not in tf.Fields:
                tf.Fields["T"] = ["volScalarField", {"dimensions": T_dims, "internalField": self.defaultTemperature_K}]
            if "p_rgh" not in tf.Fields:
                tf.Fields["p_rgh"] = ["volScalarField", {"dimensions": p_dims, "internalField": self.default_p_rgh_Pa}]
            RegionTimeFields.model_validate({"Fields": tf.Fields})

        for _, tf in self.parsed_time_solids.items():
            if "T" not in tf.Fields:
                tf.Fields["T"] = ["volScalarField", {"dimensions": T_dims, "internalField": self.defaultTemperature_K}]
            RegionTimeFields.model_validate({"Fields": tf.Fields})

        self.Time["Regions"]["fluids"] = {r: {"Fields": tf.Fields} for r, tf in self.parsed_time_fluids.items()}
        self.Time["Regions"]["solids"] = {r: {"Fields": tf.Fields} for r, tf in self.parsed_time_solids.items()}

        return self

    def check(self) -> None:
        errors = []

        # Defaults sanity
        if self.defaultTemperature_K <= 0:
            errors.append("CHTSimulation: defaultTemperature_K must be > 0 (Kelvin).")
        if self.pressureReference_Pa <= 0:
            errors.append("CHTSimulation: pressureReference_Pa must be > 0 (Pa).")

        # controlDict
        try:
            self.parsed_controlDict.check()
        except ValueError as e:
            errors.append(str(e))

        # per-region system checks
        for name, cfg in self.parsed_system_fluids.items():
            try:
                cfg.parsed_fvSchemes.check()
            except ValueError as e:
                errors.append(f"Region '{name}' (fluid): {e}")
            try:
                cfg.parsed_fvSolution.check("fluid")
            except ValueError as e:
                errors.append(f"Region '{name}' (fluid): {e}")

        for name, cfg in self.parsed_system_solids.items():
            try:
                cfg.parsed_fvSchemes.check()
            except ValueError as e:
                errors.append(f"Region '{name}' (solid): {e}")
            try:
                cfg.parsed_fvSolution.check("solid")
            except ValueError as e:
                errors.append(f"Region '{name}' (solid): {e}")

        # per-region constant checks
        for name, cfg in self.parsed_const_fluids.items():
            try:
                cfg.parsed_thermo.check()
            except ValueError as e:
                errors.append(f"Region '{name}' (fluid): {e}")

        for name, cfg in self.parsed_const_solids.items():
            try:
                cfg.parsed_thermo.check()
            except ValueError as e:
                errors.append(f"Region '{name}' (solid): {e}")

        # time fields checks
        for r, tf in self.parsed_time_fluids.items():
            if "T" not in tf.Fields or "p_rgh" not in tf.Fields:
                errors.append(f"Time/{self.Time['name']}/{r}: fluid regions must include T and p_rgh.")
        for r, tf in self.parsed_time_solids.items():
            if "T" not in tf.Fields:
                errors.append(f"Time/{self.Time['name']}/{r}: solid regions must include T.")

        if errors:
            raise ValueError("\n".join(errors))

        print("OK: semantic validation passed.")

    @property
    def controlDict(self) -> ControlDict:
        return self.parsed_controlDict

if __name__ == "__main__":
    example = {
        "System": {
            "controlDict": [
                "controlDict",
                {
                    "solverName": "chtMultiRegionFoam",
                    "startTime": 10.0,
                    "endTime": 5.0,
                    "deltaT": -0.1,
                    "writeInterval": 10
                }
            ],
            "Regions": {
                "fluids": {
                    "fluid": {
                        "fvSchemes": ["fvSchemes", {}],
                        "fvSolution": ["fvSolution", {"solvers": {}}]
                    }
                },
                "solids": {
                    "solid": {
                        "fvSchemes": ["fvSchemes", {}],
                        "fvSolution": ["fvSolution", {"solvers": {}}]
                    }
                }
            }
        },
        "Constant": {
            "Regions": {
                "fluids": {
                    "fluid": {
                        "thermophysicalProperties": [
                            "thermophysicalProperties",
                            {
                                "thermoType": {
                                    "type": "heRhoThermo",
                                    "mixture": "pureMixture",
                                    "transport": "const",
                                    "thermo": "hConst",
                                    "equationOfState": "perfectGas",
                                    "specie": "specie",
                                    "energy": "sensibleEnthalpy"
                                },
                                "mixture": {
                                    "specie": {"molWeight": 28.9},
                                    "thermodynamics": {"coeffs": {"Cp": 1000.0, "Hf": 0.0}},
                                    "transport": {"mu": 1.8e-5, "Pr": 0.7}
                                }
                            }
                        ]
                    }
                },
                "solids": {
                    "solid": {
                        "thermophysicalProperties": [
                            "thermophysicalProperties",
                            {
                                "thermoType": {
                                    "type": "heSolidThermo",
                                    "mixture": "pureMixture",
                                    "transport": "const",
                                    "thermo": "hConst",
                                    "equationOfState": "rhoConst",
                                    "specie": "specie",
                                    "energy": "sensibleEnthalpy"
                                },
                                "mixture": {
                                    "specie": {"molWeight": 50.0},
                                    "thermodynamics": {"coeffs": {"Cp": 800.0, "Hf": 0.0}},
                                    "transport": {"mu": 1e-3, "Pr": 1.0}
                                }
                            }
                        ]
                    }
                }
            }
        }
    }

    sim = CHTSimulation.model_validate(example)
    sim.check()

    print("\n=== Final Pydantic Model State Details ===")
    
    fluid_cfg = sim.parsed_system_fluids["fluid"]
    print("\n[Fluid Region] fvSchemes preset:", fluid_cfg.parsed_fvSchemes.preset)
    print("[Fluid Region] fvSolution solvers keys:", list(fluid_cfg.parsed_fvSolution.solvers.keys()))
    print("[Fluid Region] fvSolution 'T' solver:", fluid_cfg.parsed_fvSolution.solvers["T"])
    print("[Fluid Region] fvSolution 'p_rgh' solver:", fluid_cfg.parsed_fvSolution.solvers["p_rgh"])

    solid_cfg = sim.parsed_system_solids["solid"]
    print("\n[Solid Region] fvSchemes preset:", solid_cfg.parsed_fvSchemes.preset)
    print("[Solid Region] fvSolution solvers keys:", list(solid_cfg.parsed_fvSolution.solvers.keys()))
    print("[Solid Region] fvSolution 'T' solver:", solid_cfg.parsed_fvSolution.solvers["T"])