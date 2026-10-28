//! Longitudinal profiles: a grade line through points of intersection
//! (PVIs), each `(station, elevation)`, with a symmetric parabolic vertical
//! curve of a given length at each one. Grades are ratios here (rise over
//! run); the nodes show them as percentages.

/// A grade line: its PVIs in station order and the vertical curve length
/// at each (zero for none; the ends never have one).
pub struct Alignment {
    pvis: Vec<(f64, f64)>,
    lengths: Vec<f64>,
}

impl Alignment {
    /// `None` with fewer than two PVIs, or two at one station.
    pub fn new(pvis: &[[f64; 2]], lengths: &[f64]) -> Option<Self> {
        let mut paired: Vec<((f64, f64), f64)> = pvis
            .iter()
            .enumerate()
            .map(|(k, p)| ((p[0], p[1]), lengths.get(k).copied().unwrap_or(0.0).max(0.0)))
            .collect();
        paired.sort_by(|a, b| a.0 .0.total_cmp(&b.0 .0));
        if paired.len() < 2 || paired.windows(2).any(|w| w[1].0 .0 - w[0].0 .0 <= 1e-9) {
            return None;
        }
        let last = paired.len() - 1;
        let (pvis, mut lengths): (Vec<_>, Vec<_>) = paired.into_iter().unzip();
        lengths[0] = 0.0;
        lengths[last] = 0.0;
        Some(Alignment { pvis, lengths })
    }

    /// The grade of the tangent from PVI `k` to PVI `k + 1`.
    pub fn grade(&self, k: usize) -> f64 {
        let ((x0, y0), (x1, y1)) = (self.pvis[k], self.pvis[k + 1]);
        (y1 - y0) / (x1 - x0)
    }

    /// Every tangent's grade, in order.
    pub fn grades(&self) -> Vec<f64> {
        (0..self.pvis.len() - 1).map(|k| self.grade(k)).collect()
    }

    /// The vertical curve at interior PVI `k`: where it starts and ends
    /// (PVC, PVT stations) and the grades in and out.
    fn curve(&self, k: usize) -> Option<(f64, f64, f64, f64)> {
        let length = self.lengths[k];
        if length <= 0.0 || k == 0 || k + 1 >= self.pvis.len() {
            return None;
        }
        let x = self.pvis[k].0;
        Some((x - length / 2.0, x + length / 2.0, self.grade(k - 1), self.grade(k)))
    }

    /// The tangent's elevation at `x` on the line through PVI `k`.
    fn tangent(&self, k: usize, grade: f64, x: f64) -> f64 {
        let (px, py) = self.pvis[k];
        py + grade * (x - px)
    }

    /// The elevation and grade at station `x`; beyond the ends the end
    /// tangents run on.
    pub fn at(&self, x: f64) -> (f64, f64) {
        for k in 1..self.pvis.len() - 1 {
            if let Some((pvc, pvt, g1, g2)) = self.curve(k) {
                if (pvc..=pvt).contains(&x) {
                    let length = pvt - pvc;
                    let d = x - pvc;
                    let start = self.tangent(k, g1, pvc);
                    return (start + g1 * d + (g2 - g1) / (2.0 * length) * d * d, g1 + (g2 - g1) * d / length);
                }
            }
        }
        let segment = (0..self.pvis.len() - 1)
            .find(|k| x <= self.pvis[k + 1].0)
            .unwrap_or(self.pvis.len() - 2);
        let grade = self.grade(segment);
        (self.tangent(segment, grade, x), grade)
    }

    /// Stations from the first PVI to the last every `step`, with every
    /// PVI, PVC and PVT among them, in order.
    pub fn stations(&self, step: f64) -> Vec<f64> {
        let (first, last) = (self.pvis[0].0, self.pvis[self.pvis.len() - 1].0);
        let mut stations: Vec<f64> = self.pvis.iter().map(|p| p.0).collect();
        for k in 0..self.pvis.len() {
            if let Some((pvc, pvt, _, _)) = self.curve(k) {
                stations.extend([pvc, pvt]);
            }
        }
        if step > 0.0 {
            let count = (((last - first) / step) as usize).min(100_000);
            stations.extend((1..count).map(|k| first + step * k as f64));
        }
        stations.sort_by(f64::total_cmp);
        stations.dedup_by(|a, b| (*a - *b).abs() <= 1e-9);
        stations
    }

    /// Each interior PVI's curve: PVC station, PVT station and K, the length
    /// per percent of grade change (`None` where there is no curve).
    pub fn curves(&self) -> Vec<Option<(f64, f64, f64)>> {
        (0..self.pvis.len())
            .map(|k| {
                let (pvc, pvt, g1, g2) = self.curve(k)?;
                let change = ((g2 - g1) * 100.0).abs();
                Some((pvc, pvt, if change > 1e-12 { (pvt - pvc) / change } else { f64::INFINITY }))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_crest_curve_joins_its_tangents() {
        // +2 % up to station 100, then -2 %: a 100 long crest curve.
        let line = Alignment::new(&[[0.0, 10.0], [100.0, 12.0], [200.0, 10.0]], &[0.0, 100.0, 0.0]).unwrap();
        assert_eq!(line.grades(), vec![0.02, -0.02]);
        let (pvc, _) = line.at(50.0);
        assert!((pvc - 11.0).abs() < 1e-12, "on the tangent at the PVC");
        let (top, grade) = line.at(100.0);
        assert!((top - 11.5).abs() < 1e-12, "half the middle ordinate below the PVI: {top}");
        assert!(grade.abs() < 1e-12, "the high point");
        let (end, grade) = line.at(150.0);
        assert!((end - 11.0).abs() < 1e-12 && (grade + 0.02).abs() < 1e-12);
        assert_eq!(line.at(20.0), (10.4, 0.02));
        let (pvc, pvt, k) = line.curves()[1].unwrap();
        assert_eq!((pvc, pvt, k), (50.0, 150.0, 25.0));
        assert_eq!(line.stations(40.0), vec![0.0, 40.0, 50.0, 80.0, 100.0, 120.0, 150.0, 160.0, 200.0]);
        assert!(Alignment::new(&[[0.0, 1.0]], &[]).is_none());
    }
}
